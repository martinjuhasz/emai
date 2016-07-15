from emai.datasource.twitch import ChatClient, StreamClient
from emai.utils import log
from datetime import datetime
from emai.persistence import Message
from umongo import ValidationError


class Recorder(object):

    def __init__(self):
        self._chat_client = ChatClient(message_handler=self.on_chat_message, connect_handler=self.on_chat_connection)
        self._stream_client = StreamClient()

    def start(self):
        pass

    def record_channel(self, channel):
        self._chat_client.join_channel('#{}'.format(channel))

    def stop_channel(self, channel):
        self._chat_client.leave_channel('#{}'.format(channel))

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
