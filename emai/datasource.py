from emai.utils import log, config, output_stream
from livestreamer import Livestreamer, PluginError, NoPluginError, StreamError
from enum import Enum
from emai.exceptions import ResourceUnavailableException
import irc3
import re
import aiohttp


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


