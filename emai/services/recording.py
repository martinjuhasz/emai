from emai.datasource.twitch import TwitchAPI
from emai.recorder import Recorder

APP_SERVICE_KEY = 'emai_recording_service'

def setup(app):
    service = RecordingService()
    app[APP_SERVICE_KEY] = service


class RecordingService(object):
    def __init__(self):
        self.recorder = Recorder()
        self.recorder.start()

    def start_recording(self, channel):
        self.recorder.record_channel(channel)

    def stop_recording(self, channel):
        self.recorder.stop_channel(channel)