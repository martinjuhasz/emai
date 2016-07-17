from datetime import datetime
from aiohttp_utils import Response
from umongo import ValidationError
from emai import services
from emai.datasource import TwitchAPI
from emai.persistence import Recording, load_json, to_objectid, get_async_file_descriptor
from emai.utils import log
from aiohttp.web import StreamResponse

def setup(app):
    app.router.add_route('GET', '/recordings', RecorderResource.get_recordings)
    app.router.add_route('POST', '/recordings', RecorderResource.create_recording)
    app.router.add_route('PUT', '/recordings/{recording_id}/stop', RecorderResource.stop_recording)
    app.router.add_route('GET', '/recordings/{recording_id}/video', RecorderResource.get_video)

    app.router.add_route('GET', '/videos/{video_id}', VideoResource.get_video)


class Resource(object):

    @staticmethod
    async def create_video_stream_response(request):
        stream = StreamResponse()
        stream.headers['Content-Type'] = 'video/mp2t'
        stream.headers['Cache-Control'] = 'no-cache, no-store'
        stream.headers['Connection'] = 'keep-alive'
        #stream.enable_chunked_encoding()
        #await stream.prepare(request)
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

        # check if channel is already recording
        started_recording = await Recording.find_one({'channel_name': body_data['channel'], 'stopped': None})
        if started_recording:
            return Response(status=302)

        # check if channel exists on twitch
        channel_details = await TwitchAPI.get_channel_details(body_data['channel'])
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
    async def get_video(request):
        # check url parameters
        recording_id = to_objectid(request.match_info['recording_id'])
        if not recording_id:
            return Response(status=400)

        # check if recording exists
        recording = await Recording.find_one({'_id': recording_id, 'stopped': {'$exists': True}})
        if not recording:
            return Response(status=404)

        # check if video exists
        video_id = recording.video_id
        if not video_id:
            return Response(status=404)

        # generate template and return
        template = """#EXTM3U
#EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID="medium",NAME="Medium",AUTOSELECT=YES,DEFAULT=YES
#EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=992000,RESOLUTION=852x480,CODECS="avc1.66.31,mp4a.40.2",VIDEO="medium"
http://localhost:8080/videos/{video_id}
            """

        response = Response(body=bytes(template.format(video_id=video_id), encoding='UTF-8'))
        response.headers['Content-Type'] = 'application/vnd.apple.mpegurl'
        response.headers['Cache-Control'] = 'no-cache, no-store'

        return response


class VideoResource(Resource):

    @staticmethod
    async def get_video(request):
        # check url parameters
        video_id = to_objectid(request.match_info['video_id'])
        if not video_id:
            return Response(status=400)

        # start stream
        video_file_descriptor = get_async_file_descriptor()
        video_stream = await video_file_descriptor.get(video_id)

        stream_response = await RecorderResource.create_video_stream_response(request)
        stream_response.headers['Content-Length'] = str(video_stream.length)
        await stream_response.prepare(request)

        while True:
            try:
                chunk = await video_stream.read(size=4096)
                if not chunk:
                    break
                stream_response.write(chunk)
                #stream_response.write()
            except Exception as e:
                print(e)
                break

        await stream_response.write_eof()
        return stream_response
