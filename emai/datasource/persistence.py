from mongoengine import Document, EmbeddedDocument, StringField, DateTimeField, ListField, EmbeddedDocumentField, IntField, QuerySet, queryset_manager
import mongoengine
from datetime import datetime
from emai.utils import config
from enum import Enum

def connect():
    mongoengine.connect(
        config.get('persistence', 'database'),
        host=config.get('persistence', 'host', fallback='localhost'),
        port=config.getint('persistence', 'port', fallback=27017),
        username=config.get('persistence', 'username', fallback=None),
        password=config.get('persistence', 'password', fallback=None)
    )


class Emoticon(EmbeddedDocument):
    identifier = StringField(required=True)
    occurrences = ListField(StringField())


class Message(Document):
    channel_id = StringField(required=True)
    user_id = StringField(required=True)
    user_name = StringField()
    content = StringField(required=True)
    created = DateTimeField(required=True, default=datetime.utcnow())
    emoticons = ListField(EmbeddedDocumentField(Emoticon))


class Recording(Document):

    class Status(Enum):
        recording = 'recording'
        finished = 'finished'

    class RecordingQuerySet(QuerySet):
        def recording(self):
            return self.filter(status=Recording.Status.recording.value)

        def finished(self):
            return self.filter(status=Recording.Status.finished.value)

    meta = {'queryset_class': RecordingQuerySet}
    channel_name = StringField(required=True)
    channel_id = IntField(required=True)
    display_name = StringField(required=True)
    language = StringField()
    started = DateTimeField(required=True, default=datetime.utcnow())
    status = StringField(required=True, default=Status.recording.value)
