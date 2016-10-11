from datetime import datetime
from subprocess import Popen, PIPE
from bson import ObjectId
from server.twitch import ChatClient, StreamClient
from server.persistence import Message, Recording
from server.utils import log, config
from umongo import ValidationError

APP_SERVICE_KEY = 'emai_recording_service'


def setup(app, loop=None):
    service = RecordingService(loop=loop)
    app[APP_SERVICE_KEY] = service


class RecordingService(object):
    def __init__(self, loop=None):
        self.loop = loop
        self.recorder = Recorder(loop=loop)

    async def create_recording(self, channel_details):
        video_file_id = await self.start_recording(channel_details['name'])
        properties = {
            'channel_name': channel_details['name'],
            'channel_id': channel_details['_id'],
            'display_name': channel_details['display_name'],
            'language': channel_details['language'],
            'started': datetime.utcnow(),
            'video_id': str(video_file_id)
        }
        if channel_details['logo']:
            properties['logo'] = channel_details['logo']
        if channel_details['profile_banner']:
            properties['profile_banner'] = channel_details['profile_banner']
        if channel_details['video_banner']:
            properties['video_banner'] = channel_details['video_banner']
        if channel_details['profile_banner_background_color']:
            properties['background_color'] = channel_details['profile_banner_background_color']

        recording = Recording(**properties)
        await recording.commit()

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

    async def record_channel(self, channel):
        file_id = await self.record_stream(channel)
        self._chat_client.join_channel('#{}'.format(channel))
        return file_id

    async def record_stream(self, channel):
        stream = self._stream_client.get_stream(channel, StreamClient.Quality.medium)
        file_id = ObjectId()
        process = self.start_convert_process(stream, file_id)

        self.running_file_streams.append({
            'process': process,
            'file_id': file_id,
            'channel': channel
        })

        return file_id

    def start_convert_process(self, stream, file_name):
        try:
            ffmpeg_path = config.get('recording', 'ffmpeg_binary')
            video_dir = config.get('recording', 'video_dir')
            file_path = '{video_dir}/{name}.mp4'.format(video_dir=video_dir, name=file_name)
            options = '-vcodec copy -acodec copy -bsf:a aac_adtstoasc'
            command = '{ffmpeg_path} -i "{url}" {options} {name}'.format(
                ffmpeg_path=ffmpeg_path,
                video_dir=video_dir,
                options=options,
                url=stream.url,
                name=file_path)
            process = Popen(command, stdin=PIPE, shell=True)
            return process
        except Exception as e:
            print(e)

    def stop_stream_record(self, channel):
        running_records = [record for record in self.running_file_streams if record['channel'] == channel]
        for record in running_records:
            record['process'].communicate(input=bytes("q\n", encoding="UTF-8"))

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
            log.info(
                'Message saved:Time={message.created} Channel={message.channel_id} User={message.user_id}({message.username})'.format(
                    message=message))
        except ValidationError as error:
            log.warn('Message not saved: {}'.format(chat_message))
