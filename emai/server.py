import asyncio
from aiohttp import web
from emai import resources
from emai.services import recording
from emai import persistence
from aiohttp_utils import negotiation, path_norm


def create_app():
    app = web.Application(debug=True)
    resources.setup(app)  # add routing
    recording.setup(app)  # provide recorder
    negotiation.setup(app, renderers=persistence.persistence_renderer())  # automatic json responses
    path_norm.setup(app)  # normalize paths
    return app


def main():
    app = create_app()

    loop = asyncio.get_event_loop()
    handler = app.make_handler()
    srv = loop.run_until_complete(loop.create_server(handler, '0.0.0.0', 8080))
    print('serving on', srv.sockets[0].getsockname())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        loop.run_until_complete(srv.wait_closed())
        loop.run_until_complete(app.shutdown())
        loop.run_until_complete(handler.finish_connections(60.0))
        loop.run_until_complete(app.cleanup())
    loop.close()


if __name__ == "__main__":
    main()
