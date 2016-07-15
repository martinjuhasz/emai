from emai.datasource.twitch import ChatClient, StreamClient
from emai.utils import log
import datetime


class Recorder(object):

    def __init__(self, persistence):
        self.persistence = persistence
        self._chat_client = ChatClient(message_handler=self.on_chat_message, connect_handler=self.on_chat_connection)
        self._stream_client = StreamClient()

    def start(self):
        pass

    def record_channel(self, channel):
        self._chat_client.join_channel('#{}'.format(channel))

    def on_chat_connection(self):
        log.info('IRC connected.')

    async def on_chat_message(self, chat_message):
        chat_message['created'] = datetime.datetime.utcnow()
        result = await self.persistence.messages.insert(chat_message)
        log.info(('message saved', result))
