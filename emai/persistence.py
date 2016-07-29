from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFS
from bson import json_util
import gridfs
from umongo import Instance, Document, EmbeddedDocument, fields, Schema
import marshmallow
from bson import ObjectId
from bson.errors import InvalidId
import json
from functools import partial
from pymongo import MongoClient

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
        if isinstance(obj, Recording) or isinstance(obj, Bag):
            return obj.dump()
        return json.JSONEncoder.default(self, obj)


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
    data_sets = fields.ListField(fields.IntField)


@instance.register
class Bag(Document):
    class Meta:
        collection_name = 'bags'
    interval = fields.IntField(required=True)
    recording_id = fields.ObjectIdField(required=True)
    words = fields.ListField(fields.StrField, required=True)
    started = fields.DateTimeField(required=True)
    messages = fields.ListField(fields.ObjectIdField, required=True)
    message_count = fields.StrField()
    label = fields.IntField()

    @staticmethod
    def find_sample(recording_id, interval, limit=None, samples=None):
        pipeline = [
            {'$match': {
                'recording_id': recording_id,
                'interval': interval
            }},
            {'$match': {
                '$or': [{'label': {'$exists': False}}, {'label': {'$lte': 0}}]
            }},
            {'$project': {
                'recording_id': 1,
                'started': 1,
                'messages': 1,
                'words': 1,
                'interval': 1,
                'message_count': {'$size': {'$ifNull': ['$messages', []]}}
            }},
            {'$unwind': '$messages'},
            {'$lookup': {
                'from': 'messages',
                'localField': 'messages',
                'foreignField': '_id',
                'as': 'full_messages'
            }},
            {'$unwind': '$full_messages'},
            {'$group': {
                '_id': '$_id',
                'started': {'$first': '$started'},
                'video_end': {'$first': '$started'},
                'interval': {'$first': '$interval'},
                'messages': {'$first': '$messages'},
                'message_count': {'$first': '$message_count'},
                'recording_id': {'$first': '$recording_id'},
                'words': {'$first': '$words'},
                'full_messages': {"$push": "$full_messages"}
            }},

            {'$sort': {'message_count': -1}},
            {'$project': {
                'id': '$_id',
                'recording_id': 1,
                'data_set': '$interval',
                'time': '$started',
                'messages': '$full_messages',
                'words': {'$size': '$words'}
            }},
        ]
        if limit:
            pipeline.append({'$limit': limit})
        if samples:
            pipeline.append({'$sample': {'size': samples}})

        future = Bag.collection.aggregate(pipeline)
        return future

    @staticmethod
    def get_training_messages(recording_id, limit=None, label_eq={'$gte': 2}, samples=None):
        pipeline = [
            {'$match': {
                'recording_id': recording_id,
                'label': label_eq
            }},
            {'$unwind': '$messages'},
            {'$lookup': {
                'from': 'messages',
                'localField': 'messages',
                'foreignField': '_id',
                'as': 'message'
            }},
            {'$unwind': '$message'},
            {'$project': {
                '_id': 0,
                'recording_id': 1,
                'label': 1,
                'message': 1
            }},
            {'$unwind': '$message'},
            {'$unwind': '$message'},
            {'$group': {
                '_id': '$message._id',
                'recording_id': {'$first': '$recording_id'},
                'label': {'$first': '$label'},
                'channel_id': {'$first': '$message.channel_id'},
                'content': {'$first': '$message.content'},
                'created': {'$first': '$message.created'},
                'emoticons': {'$first': '$message.emoticons'},
                'user_id': {'$first': '$message.user_id'},
                'username': {'$first': '$message.username'}
            }},
            {'$project': {
                '_id': 0,
                'id': '$_id',
                'recording_id': 1,
                'label': 1,
                'channel_id': 1,
                'content': 1,
                'created': 1,
                'emoticons': 1,
                'user_id': 1,
                'username': 1
            }},

        ]
        if limit:
            pipeline.append({'$limit': limit})
        if samples:
            pipeline.append({'$sample': {'size': samples}})

        future = Bag.collection.aggregate(pipeline)
        return future

    @staticmethod
    def get_training_bags(recording_id, limit=None, label_eq={'$gte': 2}):
        pipeline = [
            {'$match': {
                'recording_id': recording_id,
                'label': label_eq
            }},
            {'$unwind': '$messages'},
            {'$lookup': {
                'from': 'messages',
                'localField': 'messages',
                'foreignField': '_id',
                'as': 'message'
            }},
            {'$unwind': '$message'},
            {'$project': {
                '_id': 0,
                'recording_id': 1,
                'label': 1,
                'message': 1
            }},
            {'$unwind': '$message'},
            {'$unwind': '$message'},
            {'$group': {
                '_id': '$message._id',
                'recording_id': {'$first': '$recording_id'},
                'label': {'$first': '$label'},
                'channel_id': {'$first': '$message.channel_id'},
                'content': {'$first': '$message.content'},
                'created': {'$first': '$message.created'},
                'emoticons': {'$first': '$message.emoticons'},
                'user_id': {'$first': '$message.user_id'},
                'username': {'$first': '$message.username'}
            }},
            {'$project': {
                '_id': 0,
                'id': '$_id',
                'recording_id': 1,
                'label': 1,
                'channel_id': 1,
                'content': 1,
                'created': 1,
                'emoticons': 1,
                'user_id': 1,
                'username': 1
            }},

        ]
        if limit:
            pipeline.append({'$limit': limit})

        future = Bag.collection.aggregate(pipeline)
        return future

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
    id = fields.ObjectIdField()
    recording_id = fields.ObjectIdField()
    words = fields.IntField()
    time = fields.DateTimeField()
    messages = fields.ListField(marshmallow.fields.Nested(MessageSchema))
    video_start = fields.IntField()
    data_set = fields.IntField()




