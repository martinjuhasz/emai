from datetime import datetime
from aiohttp_utils import Response
from umongo import ValidationError
from emai import services
from emai.datasource import TwitchAPI
from emai.persistence import Recording, Classifier, load_json, to_objectid, get_async_file_descriptor
from emai.utils import log, config
from aiohttp.web import StreamResponse
from emai.services.datasets import DataSetService
from emai.services.training import TrainingService
from emai.exceptions import ResourceExistsException, ResourceUnavailableException
import aiohttp_cors

def setup(app):
    cors = aiohttp_cors.setup(
        app,
        defaults={
            '*': aiohttp_cors.ResourceOptions(
                allow_credentials=True, expose_headers='*', allow_headers='*'
            )
        }
    )

    cors.add(app.router.add_route('GET', '/recordings', RecorderResource.get_recordings))
    cors.add(app.router.add_route('POST', '/recordings', RecorderResource.create_recording))
    cors.add(app.router.add_route('PUT', '/recordings/{recording_id}/stop', RecorderResource.stop_recording))
    cors.add(app.router.add_route('GET', '/recordings/{recording_id}/samples/{interval}', RecorderResource.samples))

    cors.add(app.router.add_route('PUT', '/messages', MessageResource.classify_messages))

    cors.add(app.router.add_route('GET', '/training/{recording_id}', TrainResource.train_classifiers))

    cors.add(app.router.add_route('GET', '/classifiers', ClassifierResource.get_classifiers))
    cors.add(app.router.add_route('POST', '/classifiers', ClassifierResource.create_classifier))
    cors.add(app.router.add_route('PUT', '/classifiers/{classifier_id}', ClassifierResource.update_classifier))
    cors.add(app.router.add_route('POST', '/classifiers/{classifier_id}/train', ClassifierResource.train_classifier))
    cors.add(app.router.add_route('GET', '/classifiers/{classifier_id}/review', ClassifierResource.review_classifier))


class Resource(object):
    @staticmethod
    async def create_video_stream_response(request):
        stream = StreamResponse()
        stream.headers['Content-Type'] = 'video/mp2t'
        stream.headers['Cache-Control'] = 'no-cache, no-store'
        stream.headers['Connection'] = 'keep-alive'
        # stream.enable_chunked_encoding()
        # await stream.prepare(request)
        return stream


class RecorderResource(Resource):
    @staticmethod
    async def get_recordings(request):
        recordings = await Recording.find().to_list(None)
        return Response(recordings)

    @staticmethod
    async def create_recording(request):
        # check if malformed request
        body_data = await load_json(request)
        if not body_data or not 'channel' in body_data:
            return Response(status=400)

        channel_name = body_data['channel'].lower()
        # check if channel is already recording
        started_recording = await Recording.find_one({'channel_name': channel_name, 'stopped': None})
        if started_recording:
            return Response(status=302)

        # check if channel exists on twitch
        channel_details = await TwitchAPI.get_channel_details(channel_name)
        if not channel_details:
            return Response(status=422)

        # start recording
        recording_service = request.app[services.recording.APP_SERVICE_KEY]
        try:
            recording = await recording_service.create_recording(channel_details)
            return Response(recording)
        except ValidationError as error:
            log.error(error)
            return Response(status=500)

    @staticmethod
    async def stop_recording(request):
        # check url parameters
        recording_id = to_objectid(request.match_info['recording_id'])
        if not recording_id:
            return Response(status=400)

        # check if recording exists
        recording = await Recording.find_one({'_id': recording_id, 'stopped': None})
        if not recording:
            return Response(status=404)

        # stop recording
        recording.stopped = datetime.utcnow()
        await recording.commit()
        recording_service = request.app[services.recording.APP_SERVICE_KEY]
        recording_service.stop_recording(recording.channel_name)

        return Response()

    @staticmethod
    async def samples(request):
        # check url parameters
        recording_id = to_objectid(request.match_info['recording_id'])
        interval = int(request.match_info['interval'])
        if not recording_id or not interval:
            return Response(status=400)

        # check if recording exists
        recording = await Recording.find_one({'_id': recording_id, 'stopped': {'$exists': True}})
        if not recording:
            return Response(status=404)

        bags = await DataSetService.get_samples(recording, interval)
        return Response(bags)


class MessageResource(Resource):

    @staticmethod
    async def classify_messages(request):

        # check if malformed request
        body_data = await load_json(request)
        if not body_data or not 'messages' in body_data:
            return Response(status=400)
        messages = body_data['messages']
        await DataSetService.classify_messages(messages)

        return Response()


class TrainResource(Resource):

    @staticmethod
    async def train_classifiers(request):
        # check url parameters
        recording_id = to_objectid(request.match_info['recording_id'])
        if not recording_id:
            return Response(status=400)

        # check if recording exists
        recording = await Recording.find_one({'_id': recording_id, 'stopped': {'$exists': True}})
        if not recording:
            return Response(status=404)

        test = await TrainingService.test(recording_id)

        return Response({
            'logreg': test['logreg'].get_results(),
            'nb': test['nb'].get_results(),
            'svm': test['svm'].get_results()
        })


class ClassifierResource(Resource):

    @staticmethod
    async def get_classifiers(request):
        classifiers = await Classifier.find().to_list(None)
        return Response(classifiers)

    @staticmethod
    async def create_classifier(request):
        # check if malformed request
        body_data = await load_json(request)
        if not body_data or not 'title' in body_data or not body_data['title'].strip():
            return Response(status=400)
        title = body_data['title'].strip()

        # create classifier
        try:
            classifier = Classifier(title=title)
            await classifier.commit()
            return Response(classifier)
        except ValidationError as error:
            log.error(error)
            return Response(status=500)

    @staticmethod
    async def update_classifier(request):
        # check url parameters
        classifier_id = to_objectid(request.match_info['classifier_id'])
        if not classifier_id:
            return Response(status=400)

        # check if recording exists
        classifier = await Classifier.find_one({'_id': classifier_id})

        # check if malformed request
        body_data = await load_json(request)
        if not body_data:
            return Response(status=400)

        try:
            await TrainingService.update_classifier(classifier, body_data)
        except ValueError as error:
            log.error(error)
            return Response(status=400)

        return Response(classifier)

    @staticmethod
    async def train_classifier(request):
        # check url parameters
        classifier_id = to_objectid(request.match_info['classifier_id'])
        if not classifier_id:
            return Response(status=400)

        # check if recording exists
        classifier = await Classifier.find_one({'_id': classifier_id})

        await TrainingService.train(classifier)

        return Response(classifier)

    @staticmethod
    async def review_classifier(request):
        # check url parameters
        classifier_id = to_objectid(request.match_info['classifier_id'])
        if not classifier_id:
            return Response(status=400)

        # check if recording exists
        classifier = await Classifier.find_one({'_id': classifier_id})

        messages = await TrainingService.review(classifier)

        return Response(messages)
