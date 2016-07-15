from aiohttp import web
from emai import persistence
from emai import recording
from aiohttp_utils import Response
from emai.persistence import load_json


def setup(app):
    app.router.add_route('*', '/recordings', RecorderResource)


class Resource(web.View):
    def get_db(self):
        return self.request.app[persistence.APP_SERVICE_KEY]

    def get_recording_service(self):
        return self.request.app[recording.APP_SERVICE_KEY]


class RecorderResource(Resource):
    async def get(self):
        db = self.get_db()
        recordings = await db.recordings.find().to_list(None)
        return Response(recordings)

    async def post(self):
        recording_service = self.get_recording_service()
        body_data = await load_json(self.request)
        if not body_data or not 'channel' in body_data:
            return Response(status=400)  # malformed request
        recorded = await recording_service.record_channel(body_data['channel'])
        if not recorded:
            return Response(status=400)  # already recording
        return Response()


