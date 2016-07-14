from emai.utils import log
from emai.datasource.twitch import ChatClient, StreamClient
from emai.datasource.persistence import Message


class Recorder(object):

    def __init__(self):
        self._chat_client = ChatClient(message_handler=self.on_chat_message)
        self._stream_client = StreamClient()

    def start(self):
        pass

    def record_channel(self, channel):
        self._chat_client.join_channel(channel)

    def on_chat_message(self, chat_message):
        message = Message(
            channel_id=chat_message['channel'],
            user_id=chat_message['user_id'],
            user_name=chat_message['username'],
            content=chat_message['content'],
            emoticons=chat_message['emoticons']
        )
        message.save()
        log.info(('message saved', message.channel_id, message.user_id, message.content))
