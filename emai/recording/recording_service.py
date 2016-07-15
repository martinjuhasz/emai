from emai.datasource.twitch import TwitchAPI
from emai.recorder import Recorder
from emai.utils import log
import datetime


class RecordingService(object):
    def __init__(self, persistence):
        self.persistence = persistence
        self.recorder = Recorder(self.persistence)
        self.recorder.start()

    async def record_channel(self, channel):
        channel_details = TwitchAPI.channel_exists(channel)
        record_exists = await self.persistence.recordings.find_one({'channel_name': channel_details['name'], 'stopped': None})
        if record_exists:
            return False

        await self.save_recording(channel_details)
        self.recorder.record_channel(channel_details['name'])
        return True

    async def save_recording(self, parameters):
        recording = {
            'channel_name': parameters['name'],
            'channel_id': parameters['_id'],
            'display_name': parameters['display_name'],
            'language': parameters['language'],
            'started': datetime.datetime.utcnow()
        }
        result = await self.persistence.recordings.insert(recording)
        return result

    def stop_recording(self, recording):
        pass