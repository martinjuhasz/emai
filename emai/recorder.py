from emai.utils import log
from emai.datasource.twitch import ChatClient, StreamClient
from emai.datasource.persistence import Message


class Recorder(object):

    def __init__(self, connect_handler=None):
        self._chat_client = ChatClient(message_handler=self.on_chat_message, connect_handler=connect_handler)
        self._stream_client = StreamClient()

    def start(self):
        self._chat_client.start()

    def on_chat_message(self, irc_message):
        message = Message(
            channel_id=irc_message['channel'],
            user_id=irc_message['user_id'],
            user_name=irc_message['username'],
            content=irc_message['content'],
            emoticons=irc_message['emoticons']
        )
        message.save()
        log.info(('message saved', message.channel_id, message.user_id, message.content))

    def record_channel(self, channel):
        self._chat_client.join_channel(channel)
