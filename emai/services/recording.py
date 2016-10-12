"""
Service-Modul für Aufgaben zur Aufnahme von Streamingkanälen.
"""
import re
from datetime import datetime
from enum import Enum
from subprocess import Popen, PIPE

import aiohttp
import irc3
from bson import ObjectId
from emai.exceptions import ResourceUnavailableException
from emai.persistence import Message, Recording
from emai.utils import log, config, output_stream
from livestreamer import Livestreamer, PluginError, NoPluginError
from umongo import ValidationError

APP_SERVICE_KEY = 'emai_recording_service'


def setup(app, loop=None):
    """
    Registriert eine Instanz des Service-Moduls beim Webserver
    """
    service = RecordingService(loop=loop)
    app[APP_SERVICE_KEY] = service


class RecordingService(object):
    def __init__(self, loop=None):
        self.loop = loop
        self.recorder = Recorder(loop=loop)

    async def create_recording(self, channel_details):
        # Recorder starten
        video_file_id = await self.start_recording(channel_details['name'])

        # Kanaldetails sammeln und Recording persistieren
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

    async def stop_recording(self, recording):
        recording.stopped = datetime.utcnow()
        await recording.commit()
        self.recorder.stop_channel(recording.channel_name)

    async def delete_recording(self, recording):
        if not recording.stopped:
            await self.stop_recording(recording)

        await Message.delete_by_recording(recording)
        await recording.remove()



class Recorder(object):
    """
    Aggregiert Video- und Chatstreams und persistiert die gewonnen Daten
    """

    def __init__(self, loop=None):
        self.loop = loop
        self._chat_client = ChatClient(message_handler=self.on_chat_message, connect_handler=self.on_chat_connection)
        self._stream_client = StreamClient()
        self.running_file_streams = []

    async def record_channel(self, channel):
        """
        Startet die Aufnahme eines Streamingkanals
        """
        file_id = await self.record_stream(channel)
        self._chat_client.join_channel('#{}'.format(channel))
        return file_id

    def stop_channel(self, channel):
        """
        Stoppt die Aufnahme eines Streamingkanals
        """
        self._chat_client.leave_channel('#{}'.format(channel))
        self.stop_stream_record(channel)

    async def record_stream(self, channel):
        """
        Startet einen Prozess zur Konvertierung des Videostreams in eine Videodatei
        """
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
        """
        Konvertiert mit ffmpeg einen Stream in eine Videodatei
        """
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
        """
        Stoppt den Prozess der Videokonvertierung
        """
        running_records = [record for record in self.running_file_streams if record['channel'] == channel]
        for record in running_records:
            record['process'].communicate(input=bytes("q\n", encoding="UTF-8"))

    def on_chat_connection(self):
        log.info('IRC connected.')

    async def on_chat_message(self, chat_message):
        """
        Persistiert eingehende Chatnachrichten
        """
        try:
            chat_message['created'] = datetime.utcnow()
            message = Message(**chat_message)
            await message.commit()
            log.info(
                'Message saved:Time={message.created} Channel={message.channel_id} User={message.user_id}({message.username})'.format(
                    message=message))
        except ValidationError as error:
            log.warn('Message not saved: {}'.format(chat_message))


class TwitchAPI(object):
    """
    Liefert Metadaten zu gewünschten Streams.
    """
    base_url = 'https://api.twitch.tv/kraken'
    headers = {
        'Accept': 'application/vnd.twitchtv.v3+json',
        'Client-ID': config.get('twitch', 'clientid')
    }

    @staticmethod
    async def make_request(url, method='GET', **kwargs):
        url = url.lstrip('/')
        kwargs.setdefault('params', {})
        kwargs.setdefault('headers', {})
        kwargs['headers'].update(TwitchAPI.headers)

        full_url = '{base_url}/{url}'.format(base_url=TwitchAPI.base_url, url=url)

        async with aiohttp.ClientSession() as session:
            async with session.request(method.upper(), full_url, **kwargs) as resposne:
                json = await resposne.json()
                return json

    @staticmethod
    async def get_channel_details(channel):
        content = await TwitchAPI.make_request('/channels/{}'.format(channel))
        return content


class StreamClient(object):
    """
    Stellt ein Video-Steaming-Objekt zur lokalen Verarbeitung zur Verfügung.
    """

    class Quality(Enum):
        source = 'source'
        high = 'high'
        medium = 'medium'
        low = 'low'
        mobile = 'mobile'

    def __init__(self):
        self._livestreamer = Livestreamer()
        self._livestreamer.set_loglevel('info')
        self._livestreamer.set_logoutput(output_stream)
        # TODO: Check oauth login instead https://www.reddit.com/r/Twitch/comments/52sye3/livestreamer_help_please_help/d7n0j36
        self._livestreamer.set_option('http-headers', 'Client-ID=jzkbprff40iqj646a697cyrvl0zt2m6')

    def get_stream(self, channel, quality):
        try:
            streams = self._livestreamer.streams('twitch.tv/{}'.format(channel))
        except NoPluginError or PluginError:
            raise ResourceUnavailableException('Could not load stream')

        if not streams:
            raise ResourceUnavailableException('Channel not found')

        if quality.value not in streams:
            raise ResourceUnavailableException('Channel quality not found')

        return streams[quality.value]


class ChatClient(object):
    """
    Stellt ein Streaming-Objekt für Chatdaten bereit.
    """

    def __init__(self, message_handler=None, connect_handler=None):
        bot_config = dict(
            nick=config.get('twitch', 'username'),
            username=config.get('twitch', 'username'),
            host=config.get('twitch', 'host'),
            port=config.getint('twitch', 'port'),
            ssl=config.getboolean('twitch', 'ssl'),
            includes=['irc3.plugins.core', __name__],
            password=config.get('twitch', 'password'),
            level='DEBUG'
        )
        self.bot = irc3.IrcBot.from_config(bot_config)
        self.bot.message_handler = message_handler
        self.bot.connect_handler = connect_handler
        self.bot.run(forever=False)

    def join_channel(self, channel):
        self.bot.join(channel)

    def leave_channel(self, channel):
        self.bot.part(channel)


@irc3.plugin
class ChatClientLoggingPlugin(object):
    """
    Plugin des ChatClients zum abgreifen und extrahieren der Chatnachrichten.
    """

    def __init__(self, context):
        self.context = context

    @irc3.event(irc3.rfc.CONNECTED)
    def connected(self, **kw):
        self.context.send('CAP REQ :twitch.tv/membership')
        self.context.send('CAP REQ :twitch.tv/tags')
        if self.context.connect_handler:
            self.context.connect_handler()

    @irc3.event(irc3.rfc.JOIN)
    def welcome(self, mask, channel, **kw):
        if channel:
            log.info('Joined channel: {}'.format(channel))

    @irc3.event(irc3.rfc.PRIVMSG)
    async def on_privmsg(self, mask=None, data=None, tags=None, **kw):
        if data and kw and tags:
            message = self.process_message(data, tags)
            if message and self.context.message_handler:
                await self.context.message_handler(message)

    @staticmethod
    def process_message(data, tagstring):
        if not data or not tagstring:
            return None

        tags = dict(tag.split('=') for tag in tagstring.split(';'))
        channel = tags.get('room-id', None)
        user_id = tags.get('user-id', None)
        username = tags.get('display-name', None)
        emoticon_string = tags.get('emotes', None)
        emoticons = [{'identifier': match[0], 'occurrences': match[1].split(',')} for match in
                     re.findall('(\d*):((?:\d*-\d*,?)+)', emoticon_string)]
        identifier = tags.get('id', None)

        if not channel or not user_id:
            return None

        return {
            'channel_id': channel,
            'user_id': user_id,
            'content': data,
            'username': username,
            'emoticons': emoticons,
            'identifier': identifier
        }
