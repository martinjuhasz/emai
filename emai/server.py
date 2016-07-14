from emai.datasource import persistence
import asyncio
from emai.service.recording_service import RecordingService
from emai.api.server import WebAPI


def main():
    # instantiate main loop
    loop = asyncio.get_event_loop()

    # setup components
    persistence.connect()

    recording_service = RecordingService()

    web_api = WebAPI(loop, recording_service)
    loop.run_until_complete(web_api.start())

    # start loop
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(web_api.shutdown())
    loop.close()


if __name__ == "__main__":
    main()
