from datetime import datetime
from umongo import ValidationError
from emai.datasource import ChatClient, StreamClient
from emai.persistence import Message, create_sync_file
from emai.utils import log
from functools import partial
from threading import Event

APP_SERVICE_KEY = 'emai_recording_service'


def setup(app, loop=None):
    service = RecordingService(loop=loop)
    app[APP_SERVICE_KEY] = service


class RecordingService(object):
    def __init__(self, loop=None):
        self.loop = loop
        self.recorder = Recorder(loop=loop)
        self.recorder.start()

    async def start_recording(self, channel):
        return await self.recorder.record_channel(channel)

    def stop_recording(self, channel):
        self.recorder.stop_channel(channel)


class Recorder(object):

    def __init__(self, loop=None):
        self.loop = loop
        self._chat_client = ChatClient(message_handler=self.on_chat_message, connect_handler=self.on_chat_connection)
        self._stream_client = StreamClient()
        self.running_file_streams = []

    def start(self):
        pass

    async def record_channel(self, channel):
        self._chat_client.join_channel('#{}'.format(channel))
        file_id = None #await self.record_stream(channel)
        return file_id

    async def record_stream(self, channel):
        stream = self._stream_client.get_stream(channel, StreamClient.Quality.medium)
        output = create_sync_file()
        stop_event = Event()
        stream_future = self.loop.run_in_executor(None, partial(StreamClient.save_stream, stream=stream, output=output, stop_event=stop_event))

        self.running_file_streams.append({
            'future': stream_future,
            'stop_event': stop_event,
            'file_id': output._id,
            'channel': channel
        })

        return output._id

    def stop_stream_record(self, channel):
        running_records = [record for record in self.running_file_streams if record['channel'] == channel]
        for record in running_records:
            record['stop_event'].set()
        # cleanup finished streams
        self.running_file_streams = [record for record in self.running_file_streams if record['future'].done()]

    def stop_channel(self, channel):
        self._chat_client.leave_channel('#{}'.format(channel))
        self.stop_stream_record(channel)

    def on_chat_connection(self):
        log.info('IRC connected.')

    async def on_chat_message(self, chat_message):
        try:
            chat_message['created'] = datetime.utcnow()
            message = Message(**chat_message)
            await message.commit()
            log.info('Message saved:Time={message.created} Channel={message.channel_id} User={message.user_id}({message.username})'.format(message=message))
        except ValidationError as error:
            log.warn('Message not saved: {}'.format(chat_message))