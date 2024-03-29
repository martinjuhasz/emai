"""
Startpunkt des Anwendungsservers. Initialisert die EventLoop und übergibt den aiohttp Webserver dieser
"""
import asyncio

from aiohttp import web
from aiohttp_utils import negotiation, path_norm
from emai import persistence
from emai import resources
from emai.services import recording
from emai.utils import config, log


def create_app(loop):
    """
    Erstellt den aiohttp Webserver und registriert Resourcen und Services bei diesem.
    :param loop: EventLoop
    :return: aiohttp Webserver
    """
    app = web.Application(debug=True, loop=loop)
    resources.setup(app)  # add routing
    recording.setup(app, loop=loop)  # provide recorder service
    negotiation.setup(app, renderers=persistence.persistence_renderer())  # automatic json responses
    path_norm.setup(app)  # normalize paths
    return app


def main():
    """
    Startet die EventLoop mit Webserver
    """
    loop = asyncio.get_event_loop()
    loop.set_debug(True)

    app = create_app(loop)
    handler = app.make_handler()

    api_host = config.get('api', 'host')
    api_port = config.getint('api', 'port')
    server = loop.run_until_complete(loop.create_server(handler, api_host, api_port))
    log.info('API started at: {host}:{port}'.format(host=api_host, port=api_port))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        loop.run_until_complete(server.wait_closed())
        loop.run_until_complete(app.shutdown())
        loop.run_until_complete(handler.finish_connections(60.0))
        loop.run_until_complete(app.cleanup())
    loop.close()


if __name__ == "__main__":
    main()
