from emai.recorder import Recorder
from emai.utils import log
from emai.datasource.twitch import TwitchAPI
from emai.datasource.persistence import Recording


class RecordingService(object):
    def __init__(self):
        self.recorder = Recorder(connect_handler=self.on_recorder_connect)
        self.recorder.start()

    def on_recorder_connect(self):
        log.info('Recorder started')
        self.record_channel('rocketbeanstv')
        self.record_channel('rocketbeanstv')

    def record_channel(self, channel):
        channel_details = TwitchAPI.channel_exists(channel)
        record_exists = Recording.objects(channel_name=channel_details['name']).recording().first()
        if record_exists:
            log.info('already recording')
            return

        record = self.save_recording(channel_details)
        self.recorder.record_channel(record.channel_name)

    @staticmethod
    def save_recording(parameters):
        recording = Recording(
            channel_name=parameters['name'],
            channel_id=parameters['_id'],
            display_name=parameters['display_name'],
            language=parameters['language']
        )
        recording.save()
        return recording

    @staticmethod
    def get_recordings():
        return Recording.objects

    def stop_recording(self, recording):
        pass