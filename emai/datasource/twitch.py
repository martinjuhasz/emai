from irc.client import Reactor
import re
from emai.utils import log, config, output_stream
from livestreamer import Livestreamer, StreamError, PluginError, NoPluginError
from enum import Enum
import requests
from emai.exceptions import ResourceUnavailableException

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
    def __init__(self, connect_handler=None, message_handler=None):
        self.reactor = Reactor()
        self.connection = self.reactor.server()
        self._connect_handler = connect_handler
        self._message_handler = message_handler
        self.reactor.add_global_handler('welcome', self.on_connect, 0)
        self.reactor.add_global_handler('pubmsg', self.on_message_event, 10)
        self.reactor.add_global_handler('disconnect', self.on_disconnect, 0)

    def start(self):
        """Start the IRC client."""
        self.connection.connect(
            config.get('twitch', 'server'),
            config.getint('twitch', 'port'),
            config.get('twitch', 'username'),
            config.get('twitch', 'password')
        )
        self.reactor.process_forever()

    def join_channel(self, channel):
        self.connection.join('#{}'.format(channel))

    def on_connect(self, connection, event):
        connection.cap('REQ', 'twitch.tv/membership')
        connection.cap('REQ', 'twitch.tv/tags')
        if self._connect_handler:
            self._connect_handler()

    def on_disconnect(self, connection, event):
        log.error(('ChatClient disconnected', event))

    def on_message_event(self, connection, event):
        message = self.parse_message(event)
        if message:
            self._message_handler(message)
        else:
            log.debug(('Unlogged message', event))

    def parse_message(self, event):
        if not event or not event.arguments or not event.tags:
            return None

        channel = next((tag['value'] for tag in event.tags if tag['key'] == 'room-id'), None)
        user_id = next((tag['value'] for tag in event.tags if tag['key'] == 'user-id'), None)
        username = next((tag['value'] for tag in event.tags if tag['key'] == 'display-name'), None)
        emoticon_string = next((tag['value'] for tag in event.tags if tag['key'] == 'emotes'), None)
        emoticons = [{'identifier': match[0], 'occurrences': match[1].split(',')} for match in
                     re.findall('(\d*):((?:\d*-\d*,?)+)', emoticon_string)] if emoticon_string else None
        identifier = next((tag['value'] for tag in event.tags if tag['key'] == 'id'), None)

        if not channel or not user_id:
            return None

        return {
            'channel': channel,
            'user_id': user_id,
            'content': event.arguments[0],
            'username': username,
            'emoticons': emoticons,
            'identifier': identifier
        }
