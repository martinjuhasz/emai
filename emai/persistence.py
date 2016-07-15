from motor.motor_asyncio import AsyncIOMotorClient
from bson import json_util
from umongo import Instance, Document, EmbeddedDocument, fields
from bson import ObjectId
import json
from functools import partial
from aiohttp.web import json_response

db = AsyncIOMotorClient()['emai']
instance = Instance(db)


def persistence_renderer():
    return {'application/json': render_json}


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
    except Exception:
        return None


class MongoJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Recording):
            return obj.dump()
        return json.JSONEncoder.default(self, obj)


@instance.register
class Emoticon(EmbeddedDocument):
    identifier = fields.StrField(required=True)
    occurrences = fields.ListField(fields.StrField())


@instance.register
class Message(Document):
    channel_id = fields.StrField(required=True)
    user_id = fields.StrField(required=True)
    username = fields.StrField()
    identifier = fields.StrField()
    content = fields.StrField(required=True)
    created = fields.DateTimeField(required=True)
    emoticons = fields.ListField(fields.EmbeddedField(Emoticon))


@instance.register
class Recording(Document):
    channel_name = fields.StrField(required=True)
    channel_id = fields.IntField(required=True)
    display_name = fields.StrField(required=True)
    language = fields.StrField()
    started = fields.DateTimeField(required=True)
    stopped = fields.DateTimeField(default=None)

