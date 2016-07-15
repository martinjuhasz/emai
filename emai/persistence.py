from motor.motor_asyncio import AsyncIOMotorClient
from bson import json_util
import json

APP_SERVICE_KEY = 'emai_persistence'


def setup(app):
    db = AsyncIOMotorClient().emai
    app[APP_SERVICE_KEY] = db
    return db


def persistence_renderer():
    return {'application/json': render_json}


def render_json(request, data):
    return json_util.dumps(data).encode('utf-8')

async def load_json(request):
    text = await request.text()
    return json_util.loads(text)