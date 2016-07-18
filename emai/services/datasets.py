from emai.persistence import Message, Bag
from emai.exceptions import ResourceUnavailableException, ResourceExistsException
from emai.utils import config
import functools
import asyncio

APP_SERVICE_KEY = 'emai_data_set_service'


def setup(app, loop=None):
    service = DataSetService(loop=loop)
    app[APP_SERVICE_KEY] = service


class DataSetService(object):
    def __init__(self, loop=None):
        self.loop = loop

    async def generate_data_set(self, recording, interval):
        if interval in recording.data_sets:
            raise ResourceExistsException

        if interval not in range(config.getint('dataset', 'interval_min'), config.getint('dataset', 'interval_max')):
            raise ValueError

        #recording.data_sets.append(interval)
        #await recording.commit()

        asyncio.ensure_future(self.start_generation(recording, interval))

    @staticmethod
    async def start_generation(recording, interval):
        interval_milli = interval * 1000
        pipeline = [
            {'$match': {
                'channel_id': str(recording.channel_id),
                'created': {'$gte': recording.started, '$lt': recording.stopped}
            }},
            {'$group': {
                '_id': {
                    '$subtract': [
                        '$created',
                        {'$mod': [{'$subtract': ['$created', recording.started]}, interval_milli]}
                    ]
                },
                'messages': {
                    '$push': '$_id'
                },
                'contents': {
                    '$push': '$content'
                }
            }}
        ]

        groups = Message.collection.aggregate(pipeline)
        async for group in groups:
            words = [token.lower() for content in group['contents'] for token in content.split(' ')]
            if not words or len(words) <= 0:
                continue

            bag = Bag(
                recording_id=recording.id,
                interval=interval,
                words=words,
                started=group['_id'],
                messages=group['messages']
            )
            await bag.commit()

        #while await groups.fetch_next:

        #bag_list = await bags.to_list(None)
        #print(len(bag_list))
        #print(bag_list)

