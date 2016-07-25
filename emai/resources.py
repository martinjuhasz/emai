from datetime import datetime
from aiohttp_utils import Response
from umongo import ValidationError
from emai import services
from emai.datasource import TwitchAPI
from emai.persistence import Recording, Bag, load_json, to_objectid, get_async_file_descriptor
from emai.utils import log, config
from aiohttp.web import StreamResponse
from emai.services.datasets import DataSetService
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
    cors.add(app.router.add_route('POST', '/recordings/{recording_id}/data-sets', RecorderResource.create_data_set))
    cors.add(app.router.add_route('GET', '/recordings/{recording_id}/data-sets/{interval}/sample',
                         RecorderResource.sample_data_set))

    cors.add(app.router.add_route('PUT', '/samples/{sample_id}',
                                  SampleResource.classify_sample))


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
        video_file_id = await recording_service.start_recording(channel_details['name'])

        # save new recording
        try:
            recording = Recording(
                channel_name=channel_details['name'],
                channel_id=channel_details['_id'],
                display_name=channel_details['display_name'],
                language=channel_details['language'],
                started=datetime.utcnow(),
                video_id=str(video_file_id)
            )
            await recording.commit()
        except ValidationError as error:
            log.error(error)
            return Response(status=500)

        return Response(recording)

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
    async def create_data_set(request):
        # check url parameters
        recording_id = to_objectid(request.match_info['recording_id'])
        if not recording_id:
            return Response(status=400)

        # check if recording exists
        recording = await Recording.find_one({'_id': recording_id, 'stopped': {'$exists': True}})
        if not recording:
            return Response(status=404)

        # check if malformed request
        body_data = await load_json(request)
        if not body_data or not 'interval' in body_data:
            return Response(status=400)

        try:
            interval = int(body_data['interval'])
            data_set_service = request.app[services.datasets.APP_SERVICE_KEY]
            await data_set_service.generate_data_set(recording, interval)
            return Response()
        except ValueError:
            return Response(status=400)
        except ResourceExistsException:
            return Response(status=409)

    @staticmethod
    async def sample_data_set(request):
        # check url parameters
        recording_id = to_objectid(request.match_info['recording_id'])
        interval = int(request.match_info['interval'])
        if not recording_id or not interval:
            return Response(status=400)

        # check if recording exists
        recording = await Recording.find_one({'_id': recording_id, 'stopped': {'$exists': True}})
        if not recording or interval not in recording.data_sets:
            return Response(status=404)

        bags = await DataSetService.get_random_samples(recording_id, interval)
        return Response(bags)


class SampleResource(Resource):

    @staticmethod
    async def classify_sample(request):
        # check url parameters
        bag_id = to_objectid(request.match_info['sample_id'])
        if not bag_id:
            return Response(status=400)

        # check if malformed request
        body_data = await load_json(request)
        if not body_data or not 'label' in body_data:
            return Response(status=400)
        label = body_data['label']
        hidden = []
        if 'hidden' in body_data:
            hidden = body_data['hidden']

        # check if bag exists and
        bag = await Bag.find_one({'_id': bag_id})
        if not bag:
            return Response(status=404)

        bag.messages = [message for message in bag.messages if str(message) not in hidden]
        bag.label = label
        await bag.commit()

        print(bag)
        return Response()




