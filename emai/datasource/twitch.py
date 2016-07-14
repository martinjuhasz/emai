from emai.utils import log, config, output_stream
from livestreamer import Livestreamer, StreamError, PluginError, NoPluginError
from enum import Enum
import requests
from emai.exceptions import ResourceUnavailableException
import irc3
import re

class TwitchAPI(object):
    base_url = 'https://api.twitch.tv/kraken'
    headers = {
        'Accept': 'application/vnd.twitchtv.v3+json',
        'Client-ID': config.get('twitch', 'clientid')
    }

    @staticmethod
    def channel_exists(channel):
        request = requests.get('{base_url}/channels/{channel}'.format(base_url=TwitchAPI.base_url, channel=channel), headers=TwitchAPI.headers)
        if not request or not request.ok or not request.json():
            raise ResourceUnavailableException('Channel not found')
        return request.json()


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


class ChatClient(object):
    def __init__(self,  message_handler=None):
        bot_config = dict(
            nick=config.get('twitch', 'username'),
            username=config.get('twitch', 'username'),
            host=config.get('twitch', 'host'),
            port=config.getint('twitch', 'port'),
            ssl=config.getboolean('twitch', 'ssl'),
            includes=['irc3.plugins.core', 'irc3.plugins.autojoins', __name__],
            autojoins=config.get('twitch', 'default_channel', fallback=[]).split(','),
            password=config.get('twitch', 'password'),
        )
        self.bot = irc3.IrcBot.from_config(bot_config)
        self.bot.message_handler = message_handler
        self.bot.run(forever=False)

    def join_channel(self, channel):
        self.bot.join(channel)


@irc3.plugin
class ChatClientLoggingPlugin(object):

    def __init__(self, context):
        self.context = context

    @irc3.event(irc3.rfc.CONNECTED)
    def connected(self, **kw):
        self.context.send('CAP REQ :twitch.tv/membership')
        self.context.send('CAP REQ :twitch.tv/tags')

    @irc3.event(irc3.rfc.JOIN)
    def welcome(self, mask, channel, **kw):
        if channel:
            log.info('Joined channel: {}'.format(channel))

    @irc3.event(irc3.rfc.PRIVMSG)
    def on_privmsg(self, mask=None, data=None, tags=None, **kw):
        if data and kw and tags:
            message = self.process_message(data, tags)
            if message:
                self.context.message_handler(message)

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
            'channel': channel,
            'user_id': user_id,
            'content': data,
            'username': username,
            'emoticons': emoticons,
            'identifier': identifier
        }


