from aiohttp import web


class WebAPI(object):
    def __init__(self, loop, recording_service):
        self.loop = loop
        self.handler = None
        self.server = None

        self.app = web.Application(loop=loop)
        self.app['recording_service'] = recording_service

        self.setup_routes()

    async def hello(self, request):
        return web.Response(body=b"Hello, world")

    async def start(self):
        self.handler = self.app.make_handler()
        self.server = await self.loop.create_server(self.handler, '0.0.0.0', 8080)

    async def shutdown(self):
        self.server.close()
        self.loop.run_until_complete(self.server.wait_closed())
        self.loop.run_until_complete(self.app.shutdown())
        self.loop.run_until_complete(self.handler.finish_connections(60.0))
        self.loop.run_until_complete(self.app.cleanup())

    def setup_routes(self):
        self.app.router.add_route('GET', '/', self.hello)


class RecorderResource(web.View):
    async def handle_intro(self, request):
        return web.Response(body=b"Hello, world")