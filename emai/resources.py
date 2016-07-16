from datetime import datetime

from aiohttp_utils import Response
from umongo import ValidationError

from emai import services
from emai.datasource import TwitchAPI
from emai.persistence import Recording
from emai.persistence import load_json, to_objectid
from emai.utils import log


def setup(app):
    app.router.add_route('GET', '/recordings', RecorderResource.get_recordings)
    app.router.add_route('POST', '/recordings', RecorderResource.create_recording)
    app.router.add_route('PUT', '/recordings/{recording_id}/stop', RecorderResource.stop_recording)


class RecorderResource(object):

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

        # check if channel is already recording
        started_recording = await Recording.find_one({'channel_name': body_data['channel'], 'stopped': None})
        if started_recording:
            return Response(status=302)

        # check if channel exists on twitch
        channel_details = await TwitchAPI.get_channel_details(body_data['channel'])
        if not channel_details:
            return Response(status=422)

        # save new recording
        try:
            recording = Recording(
                channel_name=channel_details['name'],
                channel_id=channel_details['_id'],
                display_name=channel_details['display_name'],
                language=channel_details['language'],
                started=datetime.utcnow()
            )
            await recording.commit()
        except ValidationError as error:
            log.error(error)
            return Response(status=500)

        # start recording
        recording_service = request.app[services.recording.APP_SERVICE_KEY]
        recording_service.start_recording(recording.channel_name)
        return Response()

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




