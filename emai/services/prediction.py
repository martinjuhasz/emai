"""
Service-Modul für Aufgaben zur Vorhersage von Nachrichten
"""
from emai.services.training import Trainer


class PredictionService(object):
    @staticmethod
    async def classify_messages(classifier, messages):
        """
        Klassifiziert übergebene Nachrichten mit Hilfe des Klassifikators
        """
        trainer = Trainer(classifier)
        trainer.ensure_configured()
        trainer.load()

        message_contents = [message.content for message in messages]
        predictions = trainer.estimator.predict(message_contents)
        for count, message in enumerate(messages):
            message.predicted_label = predictions[count] + 1
