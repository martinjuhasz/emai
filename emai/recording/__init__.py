from .recording_service import RecordingService
APP_SERVICE_KEY = 'emai_recording_service'
from emai import persistence


def setup(app):
    db = app[persistence.APP_SERVICE_KEY] if persistence.APP_SERVICE_KEY in app else None
    service = RecordingService(persistence=db)
    app[APP_SERVICE_KEY] = service
