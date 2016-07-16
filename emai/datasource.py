from emai.utils import log, config, output_stream
from livestreamer import Livestreamer, PluginError, NoPluginError, StreamError
from enum import Enum
from emai.exceptions import ResourceUnavailableException
import irc3
import re
import aiohttp
from functools import partial
from itertools import chain

class TwitchAPI(object):
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

    @staticmethod
    async def get_channel_streams(channel):
        content = await TwitchAPI.make_request()


class StreamClient(object):

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

    @staticmethod
    def save_stream(stream=None, output=None, stop_event=None):
        if not stream or not output:
            ResourceUnavailableException('Stream or output source not found')

        chunk_size = 1024
        stream_fd = None
        try:
            stream_fd, prebuffer = StreamClient.open_stream(stream)
            stream_iterator = chain(
                [prebuffer],
                iter(partial(stream_fd.read, chunk_size), b"")
            )
            log.info('Stream recording started.')
            for data in stream_iterator:
                if stop_event.is_set():
                    break
                output.write(data)
        except ValueError or TypeError as error:
            log.error('Error writing stream: {}'.format(error))
        finally:
            if stream_fd:
                stream_fd.close()
            log.info('Stream recording stopped.')

    @staticmethod
    def open_stream(stream):
        try:
            stream_fd = stream.open()
        except StreamError as error:
            raise ResourceUnavailableException('Could not open stream: {0}'.format(error))

        try:
            prebuffer = stream_fd.read(8192)
        except IOError as err:
            raise ResourceUnavailableException('Failed to read data from stream: {0}'.format(err))

        if not prebuffer:
            raise ResourceUnavailableException('No data returned from stream')

        return stream_fd, prebuffer


class ChatClient(object):
    def __init__(self,  message_handler=None, connect_handler=None):
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
        emoticons = [{'identifier': match[0], 'occurrences': match[1].split(',')} for match in re.findall('(\d*):((?:\d*-\d*,?)+)', emoticon_string)]
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


