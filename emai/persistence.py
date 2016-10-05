from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFS
from bson import json_util
import gridfs
from umongo import Instance, Document, EmbeddedDocument, fields, Schema, BaseSchema
from marshmallow import fields as marshmallow_fields
from umongo.abstract import BaseField
import marshmallow
from bson import ObjectId
from bson.errors import InvalidId
import json
from functools import partial
from pymongo import MongoClient
import re
from marshmallow import fields as ma_fields
import datetime

db = AsyncIOMotorClient()['emai']
instance = Instance(db)


def persistence_renderer():
    return {
        'application/json': render_json
    }


def render_json(request, data):
    dumps = partial(json.dumps, cls=MongoJsonEncoder, indent=True)
    json_string = dumps(data)
    return json_string.encode('utf-8')


class MethodField(BaseField, marshmallow_fields.Method):
    pass

async def load_json(request):
    text = await request.text()
    return json_util.loads(text)


def to_objectid(id_string):
    try:
        return ObjectId(id_string)
    except InvalidId or TypeError:
        return None


def create_sync_file():
    database = MongoClient().emai
    grid_fs = gridfs.GridFS(database)
    file = grid_fs.new_file()
    return file


async def create_async_file():
    grid_fs = AsyncIOMotorGridFS(db)
    file = await grid_fs.new_file()
    return file


def get_async_file_descriptor():
    grid_fs = AsyncIOMotorGridFS(db)
    return grid_fs


class MongoJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # TODO: Why does this not work? => if isinstance(obj, Document):
        if isinstance(obj, Recording) or isinstance(obj, Classifier) or isinstance(obj, Message):
            return obj.dump()
        return json.JSONEncoder.default(self, obj)


class BytesField(BaseField, ma_fields.String):
    def _deserialize(self, value, attr, data):
        if isinstance(value, bytes):
            return value
        return super()._deserialize(value, attr, data)


@instance.register
class PerformanceResult(EmbeddedDocument):
    time = fields.ListField(fields.DateTimeField())
    precision = fields.ListField(fields.FloatField())
    recall = fields.ListField(fields.FloatField())
    fscore = fields.ListField(fields.FloatField())
    support = fields.ListField(fields.IntegerField())


@instance.register
class Performance(EmbeddedDocument):
    positive = fields.EmbeddedField(PerformanceResult)
    negative = fields.EmbeddedField(PerformanceResult)
    neutral = fields.EmbeddedField(PerformanceResult)

    @staticmethod
    def create_empty_performance():
        performance = Performance()
        performance.positive = PerformanceResult()
        performance.negative = PerformanceResult()
        performance.neutral = PerformanceResult()
        return performance


@instance.register
class Classifier(Document):
    class Meta:
        collection_name = 'classifiers'
    title = fields.StrField(required=True)
    training_sets = fields.ListField(fields.ObjectIdField())
    test_set = fields.ListField(fields.ObjectIdField(), load_only=True, dump_only=True)
    train_set = fields.ListField(fields.ObjectIdField(), load_only=True, dump_only=True)
    unlabeled_train_set = fields.ListField(fields.ObjectIdField())
    type = fields.IntegerField()
    settings = fields.DictField()
    state = BytesField(load_only=True, dump_only=True)
    performance = fields.EmbeddedField(Performance)

    def has_train_set(self):
        if self.train_set and len(self.train_set) >= 0:
            return True
        return False


    async def reset(self):
        reset_fields = ['state', 'performance', 'train_set', 'unlabeled_train_set']
        await Classifier.collection.update({'_id': self.id}, {'$unset': {field: '' for field in reset_fields}})
        for field in reset_fields:
            if hasattr(self, field) and getattr(self, field):
                delattr(self, field)


@instance.register
class Emoticon(EmbeddedDocument):
    identifier = fields.StrField(required=True)
    occurrences = fields.ListField(fields.StrField())


@instance.register
class Message(Document):
    class Meta:
        collection_name = 'messages'

    channel_id = fields.StrField(required=True)
    user_id = fields.StrField(required=True)
    username = fields.StrField()
    identifier = fields.StrField()
    content = fields.StrField(required=True)
    created = fields.DateTimeField(required=True)
    emoticons = fields.ListField(fields.EmbeddedField(Emoticon))
    label = fields.IntField()
    predicted_label = fields.IntField()

    @staticmethod
    async def get_random(channel_filter, label=None, amount=1):
        if label is None:
            label_filter = {}
        elif label is False:
            label_filter = {'label': {'$exists': False}}
        else:
            label_filter = {'label': label}

        pipeline = [
            {'$match': {**{'$or': channel_filter}, **label_filter}},
            {'$sample': {'size': amount}}
        ]
        aggregated_messages = await Message.collection.aggregate(pipeline).to_list(None)
        messages = [Message.build_from_mongo(docs) for docs in aggregated_messages]
        return messages

    @staticmethod
    async def at_time(recording, until_time, last_message=None):
        max_datetime = recording.started + datetime.timedelta(seconds=until_time)
        channel_filter = {'channel_id': str(recording.channel_id)}
        created_filter = {'created': {'$lt': max_datetime }}
        id_filter = {}
        if last_message:
            id_filter = {'id': {'$gt': last_message}}
        else:
            min_datetime = recording.started + datetime.timedelta(seconds=max(until_time - 30, 0))
            created_filter = {'created': {'$lt': max_datetime, '$gte': min_datetime}}
        filters = {**channel_filter, **created_filter, **id_filter}

        future = Message.find(filters).sort('_id', 1).limit(300)
        return await future.to_list(None)

    @staticmethod
    async def find_random(query, amount=None):
        pipeline = [{'$match': query}]
        if amount:
            pipeline.append({'$sample': {'size': amount}})

        raw_docs = await Message.collection.aggregate(pipeline).to_list(None)
        docs = [Message.build_from_mongo(docs) for docs in raw_docs]
        return docs

    @staticmethod
    def find_sample(recording, interval, limit=None, samples=None):
        interval_milli = interval * 1000
        pipeline = [
            {'$match': {
                'channel_id': str(recording.channel_id),
                'created': {'$gte': recording.started, '$lt': recording.stopped},
                'content': {'$not': re.compile('^[!|@].*')},
                'label': {'$exists': False}
            }},
            {'$group': {
                '_id': {
                    '$subtract': [
                        '$created',
                        {'$mod': [{'$subtract': ['$created', recording.started]}, interval_milli]}
                    ]
                },
                'video_start': {'$first': '$created'},
                'messages': {
                    '$push': '$_id'
                }
            }},
            {'$unwind': '$messages'},
            {'$lookup': {
                'from': 'messages',
                'localField': 'messages',
                'foreignField': '_id',
                'as': 'messages'
            }},
            {'$unwind': '$messages'},
            {'$group': {
                '_id': '$_id',
                'video_start': {'$first': '$video_start'},
                'messages': {"$push": "$messages"}
            }},
            {'$project': {
                '_id': 0,
                'id': '$_id',
                'video_start': {'$divide': [{'$subtract': ['$video_start', recording.started]}, 1000]},
                'messages': 1
            }},
        ]
        if limit:
            pipeline.append({'$limit': limit})
        if samples:
            pipeline.append({'$sample': {'size': samples}})

        future = Message.collection.aggregate(pipeline)
        return future

    @staticmethod
    def clusters(recording, interval, limit=None):
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
                }
            }},
            {'$project': {
                '_id': 0,
                'id': '$_id',
                'messages': {'$size': "$messages"}
            }},
            {'$sort': {'messages': -1}}
        ]
        if limit:
            pipeline.append({'$limit': limit})

        future = Message.collection.aggregate(pipeline)
        return future


@instance.register
class Recording(Document):
    class Meta:
        collection_name = 'recordings'

    channel_name = fields.StrField(required=True)
    channel_id = fields.IntField(required=True)
    display_name = fields.StrField(required=True)
    language = fields.StrField()
    started = fields.DateTimeField(required=True)
    stopped = fields.DateTimeField()
    video_id = fields.StrField()
    logo = fields.StrField()
    profile_banner = fields.StrField()
    video_banner = fields.StrField()
    background_color = fields.StrField()

    async def get_stats(self):
        message_filter = {'channel_id': str(self.channel_id), 'created': {'$gte': self.started, '$lt': self.stopped}}
        total_messages = await Message.find(message_filter).count()
        positive_messages = await Message.find({**message_filter, **{'label': 3}}).count()
        negative_messages = await Message.find({**message_filter, **{'label': 2}}).count()
        neutral_messages = await Message.find({**message_filter, **{'label': 1}}).count()
        return {
            'total_messages': total_messages,
            'positive_messages': positive_messages,
            'negative_messages': negative_messages,
            'neutral_messages': neutral_messages
        }

class EmoticonSchema(Schema):
    identifier = fields.StrField(required=True)
    occurrences = fields.ListField(fields.StrField())


class MessageSchema(Schema):
    _id = fields.ObjectIdField()
    channel_id = fields.StrField(required=True)
    user_id = fields.StrField(required=True)
    username = fields.StrField()
    identifier = fields.StrField()
    content = fields.StrField(required=True)
    created = fields.DateTimeField(required=True)
    emoticons = fields.ListField(marshmallow.fields.Nested(EmoticonSchema))


class SampleSchema(Schema):
    id = fields.DateTimeField()
    messages = fields.ListField(marshmallow.fields.Nested(MessageSchema))
    video_start = fields.IntegerField()

