from bson import ObjectId
from bson.errors import InvalidId
from emai.persistence import Message, SampleSchema
from emai.utils import config, log

APP_SERVICE_KEY = 'emai_message_service'


def setup(app, loop=None):
    service = MessageService(loop=loop)
    app[APP_SERVICE_KEY] = service


class MessageService(object):
    def __init__(self, loop=None):
        self.loop = loop

    @staticmethod
    async def get_samples(recording, interval):
        pool_size = config.getint('training', 'random_sample_pool')
        sample_size = config.getint('training', 'random_sample_size')
        samples_future = Message.find_sample(recording, interval, limit=pool_size, samples=sample_size).to_list(None)
        samples = []
        schema = SampleSchema()
        for data in await samples_future:
            samples.append(schema.dump(data).data)
        return samples

    @staticmethod
    async def classify_messages(messages):
        for message_data in messages:
            try:
                message_id = ObjectId(message_data['id'])
                message = await Message.find_one(message_id)
                if not message:
                    continue
                message.label = message_data['label']
                await message.commit()
                log.info('Message classified: {} label:{}'.format(message.id, message.label))
            except TypeError or InvalidId:
                log.warn('Could not save classification: {}'.format(message_data))
