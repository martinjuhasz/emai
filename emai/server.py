from emai.datasource import persistence
from emai.service.recording_service import RecordingService
from emai.api.server import APIServer

def main():
    persistence.connect()

    recording_service = RecordingService()

    api_server = APIServer()
    api_server.start()


if __name__ == "__main__":
    main()
