"""
Dieses Service-Modul stellt alle Funktionalitäten für das Konfigurieren und Trainieren eines Klassifikators bereit
"""
import math
import pickle
import re
from datetime import datetime
from enum import Enum
from random import randint
from random import shuffle

import numpy as np
import scipy
from bson import ObjectId
from emai.persistence import Message, Recording, Performance, Classifier
from emai.utils import log
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class ClassifierType(Enum):
    NaiveBayes = 1
    SupportVectorMachine = 2
    LogisticRegression = 3


class LearnType(Enum):
    LeastConfident = 1
    MostInformative = 2


class TrainingService(object):
    @staticmethod
    async def update_classifier(classifier, new_data):
        """
        Aktualisieren der Klassifikator-Einstellungen
        """
        if 'training_sets' in new_data:
            classifier.training_sets = [ObjectId(r_id) for r_id in new_data['training_sets']]

        if 'type' in new_data and ClassifierType(new_data['type']):
            classifier.type = new_data['type']

        if 'settings' in new_data:
            if any(setting in new_data['settings'] for setting in ('ngram_range', 'stop_words', 'idf', 'c', 'alpha', 'gamma')):
                classifier.settings = new_data['settings']

        # Erstellung eines neuen Testsets
        data_source = DataSource(classifier)
        classifier.test_set = await data_source.generate_test_data()

        await classifier.commit()
        await classifier.reset()

    @staticmethod
    async def delete_classifier(classifier):
        await classifier.delete()

    @staticmethod
    async def train(classifier, train_count=None):
        """
        Trainieren eines Klassifikators mit zufälligen Trainingsdaten
        """
        trainer = Trainer(classifier)
        trainer.ensure_configured()

        await trainer.train_until(train_count)

    @staticmethod
    async def learn(classifier):
        """
        Trainieren eines Klassifikators mit Active Learning
        """
        trainer = Trainer(classifier)
        trainer.ensure_configured()

        await trainer.learn()

        return await trainer.messages_for_mentoring()


class PreProcessing(object):
    stop_words = ['aber', 'als', 'am', 'an', 'auch', 'auf', 'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'dadurch',
                  'daher', 'darum', 'das', 'daß', 'dass', 'dein', 'deine', 'dem', 'den', 'der', 'des', 'dessen',
                  'deshalb', 'die', 'dies', 'dieser', 'dieses', 'doch', 'dort', 'du', 'durch', 'ein', 'eine', 'einem',
                  'einen', 'einer', 'eines', 'er', 'es', 'euer', 'eure', 'für', 'hatte', 'hatten', 'hattest', 'hattet',
                  'hier', 'hinter', 'ich', 'ihr', 'ihre', 'im', 'in', 'ist', 'ja', 'jede', 'jedem', 'jeden', 'jeder',
                  'jedes', 'jener', 'jenes', 'jetzt', 'kann', 'kannst', 'können', 'könnt', 'machen', 'mein', 'meine',
                  'mit', 'muß', 'mußt', 'musst', 'müssen', 'müßt', 'nach', 'nachdem', 'nein', 'nicht', 'nun', 'oder',
                  'seid', 'sein', 'seine', 'sich', 'sie', 'sind', 'soll', 'sollen', 'sollst', 'sollt', 'sonst',
                  'soweit', 'sowie', 'und', 'unser', 'unsere', 'unter', 'vom', 'von', 'vor', 'wann', 'warum', 'was',
                  'weiter', 'weitere', 'wenn', 'wer', 'werde', 'werden', 'werdet', 'weshalb', 'wie', 'wieder', 'wieso',
                  'wir', 'wird', 'wirst', 'wo', 'woher', 'wohin', 'zu', 'zum', 'zur', 'über', "a", "about", "above",
                  "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be",
                  "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot",
                  "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
                  "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
                  "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                  "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
                  "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
                  "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
                  "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
                  "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
                  "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
                  "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
                  "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
                  "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
                  "you're", "you've", "your", "yours", "yourself", "yourselves"]

    @staticmethod
    def max_feature_count(count_vectorizer, min_occurences, train_data):
        vocabulary_matrix = count_vectorizer.fit_transform(train_data)
        vocabulary_count = np.asarray(vocabulary_matrix.sum(axis=0)).ravel()
        excluded = np.sum(i > min_occurences for i in vocabulary_count)
        return len(vocabulary_count) - excluded


class Trainer(object):
    """
    Erstellt und kontrolliert die Trainingsvorgänge eines Klassifikatorobjektes.
    """
    def __init__(self, classifier):
        self.classifier = classifier

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value
        self.estimator = None
        self.test_data = None
        self.test_target = None
        if value:
            self.datasource = DataSource(value)
        else:
            self.datasource = None

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    def reset(self):
        if self.classifier:
            self.datasource = DataSource(self.classifier)

    def ensure_configured(self):
        if not self.classifier.settings:
            raise ValueError('Classifier Settings must be set')
        if 'ngram_range' not in self.classifier.settings:
            raise ValueError('Classifier Settings ngram_range must be set')
        if 'stop_words' not in self.classifier.settings:
            raise ValueError('Classifier Settings stop_words must be set')
        if 'idf' not in self.classifier.settings:
            raise ValueError('Classifier Settings idf must be set')

        classifier_type = ClassifierType(self.classifier.type)
        if classifier_type == ClassifierType.LogisticRegression and not 'c' in self.classifier.settings:
            raise ValueError('Classifier Settings C parameter must be set')
        if classifier_type == ClassifierType.SupportVectorMachine and not 'c' in self.classifier.settings:
            raise ValueError('Classifier Settings C parameter must be set')
        if classifier_type == ClassifierType.SupportVectorMachine and not 'gamma' in self.classifier.settings:
            raise ValueError('Classifier Settings Gamma parameter must be set')
        if classifier_type == ClassifierType.NaiveBayes and not 'alpha' in self.classifier.settings:
            raise ValueError('Classifier Settings Alpha parameter must be set')

    async def ensure_train_set(self):
        if self.classifier.has_train_set():
            return
        self.classifier.train_set = await self.datasource.get_sample_train_set()

    async def train(self):
        """
        Trainiert einen Scitit-Klassifikator anhand der des Klassifikatorobjekts aus der Datenbank
        """
        self.ensure_configured()
        await self.ensure_train_set()

        # Trainingsdaten laden
        train_data, train_target = await self.datasource.load_train_data()

        # Pipeline erstellen
        ngram_range = (1, self.classifier.settings['ngram_range'])
        stop_words = PreProcessing.stop_words if self.classifier.settings['stop_words'] else None
        count_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        tfidf_tranformer = TfidfTransformer(use_idf=self.classifier.settings['idf'])
        estimator = self.choose_estimator()
        pipeline = Pipeline([('vect', count_vectorizer), ('tfidf', tfidf_tranformer), ('cls', estimator)])

        # Klassifikator trainieren und in Property sichern
        pipeline.fit(train_data, train_target)
        self.estimator = pipeline

    def choose_estimator(self):
        """
        Gibt den passenden Klassifikator anhand der Einstellungen zurück
        """
        classifier_type = ClassifierType(self.classifier.type)
        if classifier_type == ClassifierType.LogisticRegression:
            c_param = self.get_free_param('c')
            return LogisticRegression(random_state=42, C=c_param)
        elif classifier_type == ClassifierType.SupportVectorMachine:
            c_param = self.get_free_param('c')
            gamma_param = self.get_free_param('gamma')
            return SVC(random_state=10, C=c_param, gamma=gamma_param)
        elif classifier_type == ClassifierType.NaiveBayes:
            alpha_param = self.get_free_param('alpha')
            return MultinomialNB(alpha=alpha_param)

    def get_free_param(self, param):
        if param == 'c':
            if 'c' not in self.classifier.settings:
                return 2
            elif self.classifier.settings == 1:
                return 0.25
            elif self.classifier.settings == 2:
                return 0.5
            elif self.classifier.settings == 3:
                return 1
            elif self.classifier.settings == 4:
                return 2
            elif self.classifier.settings == 5:
                return 4
            else:
                return 2
        elif param == 'gamma':
            if 'gamma' not in self.classifier.settings:
                return 0.5
            elif self.classifier.settings == 1:
                return 'auto'
            elif self.classifier.settings == 2:
                return 0.01
            elif self.classifier.settings == 3:
                return 0.1
            elif self.classifier.settings == 4:
                return 0.25
            elif self.classifier.settings == 5:
                return 0.5
            elif self.classifier.settings == 6:
                return 0.75
            else:
                return 0.5
        elif param == 'alpha':
            if 'alpha' not in self.classifier.settings:
                return 0.5
            elif self.classifier.settings == 1:
                return 0.25
            elif self.classifier.settings == 2:
                return 0.5
            elif self.classifier.settings == 3:
                return 1
            elif self.classifier.settings == 4:
                return 2
            elif self.classifier.settings == 5:
                return 4
            else:
                return 0.5

    async def save(self):
        """
        Speichert den Zustand eines trainierten Scikit Klassifikator im Datenbankobjekt ab
        """
        classifier_state = pickle.dumps(self.estimator)
        self.classifier.state = classifier_state
        await self.classifier.commit()

    def load(self):
        """
        Lädt den Zustand eines Scikit Klassifikator vom Datenbankobjekt
        """
        if self.classifier.state and not self.estimator:
            self.estimator = pickle.loads(self.classifier.state)

    def is_waiting_for_mentoring(self):
        """
        Gibt zurück ob ein Klassifikator noch ungelabelte Chatnachrichten seitens des Orkakels besitzt
        """
        if self.classifier.unlabeled_train_set and len(self.classifier.unlabeled_train_set) >= 0:
            return True
        return False

    def add_for_mentoring(self, message_id):
        """
        Fügt Nachrichten dem Stack der zu labelnden Chatnachrichten durch das Orakel hinzu
        """
        self.classifier.unlabeled_train_set.append(message_id)

    async def check_for_mentoring(self):
        """
        Prüft ob zu labelnde Chatnachrichten zwischenzeitlich klassifiziert wurden und bewegt diese ins Trainingsset.
        """
        newly_labeled = []
        for message_id in self.classifier.unlabeled_train_set:
            message = await Message.find_one({'id': message_id})
            if message.label and message.label > 0:
                newly_labeled.append(message_id)
                self.classifier.train_set.append(message_id)
        new_unlabeled_set = [mid for mid in self.classifier.unlabeled_train_set if mid not in newly_labeled]
        self.classifier.unlabeled_train_set = new_unlabeled_set

    async def train_until(self, train_count, test_iterative=False):
        """
        Trainiert einen Klassifikator bis zu einer bestimmten Trainingsmengengröße
        """
        if not train_count:
            train_count = await self.datasource.get_max_train_count()

        current_train_count = len(self.classifier.train_set)
        if train_count <= current_train_count:
            return

        add_train_set = await self.datasource.get_sample_train_set(amount=train_count - current_train_count)
        # Nach jedem einzelnen Sample testen
        if test_iterative:
            for train_set in add_train_set:
                self.classifier.train_set.append(train_set)
                try:
                    await self.train()
                    await self.test()
                except ValueError:
                    pass
        else:
            self.classifier.train_set.extend(add_train_set)
            await self.train()
            await self.test()

        await self.save()

    async def learn(self, test=True, save=True, max_learn_count=None, randomize=False, randomize_step=False,
                    interactive=True, learn_type=LearnType.LeastConfident):
        """
        Trainiert einen Klassifikator mit Hilfe von Active Learning bis Nachrichten vom Orakel klassifiziert werden müssen
        :param test: Soll der Klassifikator getestet werden
        :param save: Soll der Zustand gespeichert werden
        :param max_learn_count: maximale Anzahl an Trainingsdaten
        :param randomize: Sollen zufällige Trainingsdaten eingestreut werden
        :param randomize_step: Wie oft sollen zufällige Trainingsdaten eingestreut werden
        :param interactive: Sollen auch ungelabelte Trainingsdaten miteinbezogen werden
        :param learn_type: Typ des Active Learning Algorithmus: default: LeastConfident -> Uncertainty Sampling
        """
        # Abbrechen wenn noch ungelabelte Chatnachrichten vorliegen
        await self.check_for_mentoring()
        if self.is_waiting_for_mentoring():
            return

        await self.train()
        if test:
            await self.test()

        # iterativ trainieren solange gelabelte Trainingsdaten existieren
        randomize_counter = 1
        while not self.is_waiting_for_mentoring():
            # Abruch wenn maximale Trainingsgrenze erreicht ist
            if max_learn_count and len(self.classifier.train_set) > max_learn_count:
                break

            # Active Learning anwenden, wenn kein randomize aktiviert ist
            if not randomize or (randomize and randomize_step and randomize_counter % randomize_step == 0):
                if learn_type == LearnType.LeastConfident:
                    next_message = await self.datasource.least_confident_message(self.estimator,
                                                                                 interactive=interactive)
                else:
                    next_message = await self.datasource.most_informative(interactive=interactive)
            # ansonsten zufällige Nachricht auswählen
            else:
                next_message = await self.datasource.random_message(interactive=interactive)

            # Abbruch wenn es keine Nachrichten mehr gibt
            if not next_message:
                break

            # Nachricht dem Orakel vorlegen falls kein Label existiert
            if not next_message.label:
                self.add_for_mentoring(next_message.id)
                continue

            # Ansonsten Nachricht dem Trainingsset hinzufügen und trainieren
            randomize_counter += 1
            self.classifier.train_set.append(next_message.id)
            await self.train()
            if test:
                await self.test()

        if save:
            await self.save()

    async def messages_for_mentoring(self):
        """
        Gibt alle Nachrichten zurück die noch vom Orkal klassifiziert werden müssen.
        """
        if not self.is_waiting_for_mentoring():
            return None

        cursor = Message.find({'_id': {'$in': self.classifier.unlabeled_train_set}})
        return await cursor.to_list(None)

    async def test(self):
        """
        Testet den Klassifikator auf verschiedene Scores und speichert diese im Klassifikatorobjekt
        :return:
        """
        if not self.test_data or not self.test_target:
            self.test_data, self.test_target = await self.datasource.load_test_data()

        predicted = self.estimator.predict(self.test_data)
        precision, recall, fscore, support = precision_recall_fscore_support(self.test_target, predicted)

        time = datetime.utcnow()
        if not self.classifier.performance:
            self.classifier.performance = Performance.create_empty_performance()
        self.classifier.performance._modified = True  # TODO: is this a bug? shouldn't have to do that: report

        self.classifier.performance.neutral.time.append(time)
        self.classifier.performance.neutral.precision.append(precision[0].item())
        self.classifier.performance.neutral.recall.append(recall[0].item())
        self.classifier.performance.neutral.fscore.append(fscore[0].item())
        self.classifier.performance.neutral.support.append(support[0].item())

        self.classifier.performance.negative.time.append(time)
        self.classifier.performance.negative.precision.append(precision[1].item())
        self.classifier.performance.negative.recall.append(recall[1].item())
        self.classifier.performance.negative.fscore.append(fscore[1].item())
        self.classifier.performance.negative.support.append(support[1].item())

        self.classifier.performance.positive.time.append(time)
        self.classifier.performance.positive.precision.append(precision[2].item())
        self.classifier.performance.positive.recall.append(recall[2].item())
        self.classifier.performance.positive.fscore.append(fscore[2].item())
        self.classifier.performance.positive.support.append(support[2].item())

    async def score(self):
        """
        Trifft eine Vorhersage für Trainings- und Testdaten des Klassifikators
        :return:
        """
        if not self.test_data or not self.test_target:
            self.test_data, self.test_target = await self.datasource.load_test_data()
        train_data, train_target = await self.datasource.load_train_data()

        scores = [self.estimator.score(train_data, train_target),
                  self.estimator.score(self.test_data, self.test_target)]
        return scores


class DataSource(object):
    """
    Stellt die Datenquelle für alle Operationen mit Klassifikatoren bereit. Nutzt die Persistenzschicht zur Aggregation
    der Daten und formt diese zur Verwendung mit den scikit Klassen um.
    """
    def __init__(self, classifier):
        self.classifier = classifier
        self.informative_stack = None

    @staticmethod
    def entropy(data):
        if data is None:
            return 0
        entropy = 0
        for p_x in data:
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
        return entropy

    async def get_sample_train_set(self, amount=9):
        """
        Stellt ein Startset an Trainingsdaten bereit
        """
        amount = math.ceil(amount / 3)
        channel_filter = await self.create_channel_filter()
        positive = await Message.get_random(channel_filter, 3, amount)
        negative = await Message.get_random(channel_filter, 2, amount)
        neutral = await Message.get_random(channel_filter, 1, amount)
        ids = [message.id for message in positive + negative + neutral]
        return ids

    async def get_max_train_count(self):
        """
        Zählt die verfügbaren Chatnachrichten mit Labeln
        """
        channel_filter = await self.create_channel_filter()
        message_filter = {**{'$or': channel_filter}, **{'label': {'$exists': True}}}
        return await Message.find(message_filter).count()

    async def random_message(self, interactive=False):
        channel_filter = await self.create_channel_filter()
        label = randint(1, 3) if not interactive else None
        messages = await Message.get_random(channel_filter, label)
        return messages[0]

    async def generate_informative_messages(self, interactive=False):
        """
        Generiert eine Liste mit Chatnachrichten die häufig genutzt Wörter enthalten.
        """
        # load vocabulary data
        vocabulary_cursor = await self.generate_vocabulary_data(include_unlabeled=interactive)
        vocabulary_data, vocabulary_target = await self.load_data(vocabulary_cursor)

        # setup count vectorizer
        ngram_range = (1, self.classifier.settings['ngram_range'])
        stop_words = PreProcessing.stop_words if self.classifier.settings['stop_words'] else None
        count_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)

        # transform to sorted vocabulary
        vocabulary_matrix = count_vectorizer.fit_transform(vocabulary_data)
        vocabulary = sorted(
            list(zip(count_vectorizer.get_feature_names(), np.asarray(vocabulary_matrix.sum(axis=0)).ravel().tolist())),
            key=lambda item: item[1], reverse=True)

        # search for messages with top 10 vocabulary
        top_searches = [re.compile(token, re.I) for token, token_count in vocabulary[:10]]
        channel_filter = await self.create_channel_filter()
        label_filter = {} if interactive else {'label': {'$exists': True}}
        message_cursor = Message.find({**{'$or': channel_filter, 'content': {'$in': top_searches}}, **label_filter})
        self.informative_stack = message_cursor

    async def most_informative(self, interactive=False):
        """
        Gibt eine Nachricht zurück die Wörter enthält die häufig im Dokumentenraum vorkommen.
        Wurde nicht evaluiert oder in der Arbeit erwähnt.
        """
        if not self.informative_stack:
            await self.generate_informative_messages(interactive=interactive)

        if await self.informative_stack.fetch_next:
            msg = self.informative_stack.next_object()
            return msg
        else:
            return None

    async def least_confident_message(self, estimator, interactive=False):
        """
        Gibt eine Nachricht mit hohem Informationsgehalt zurück, ermittelt durch den höchsten
        Entropiewert von Vorhersagen aller Chatnachrichten des gesetzten Klassifikators
        """

        # Chatnachrichten die nicht bereits in Test- und Trainingsmenge enthalten sind
        # messages = Chatnachrichten der recordings des klassifikators - test_set - train_set - unlabeled_train_set
        messages = await self.generate_review_data(include_unlabeled=interactive)
        messages_data = [message.content for message in messages]

        # Vorhersagen auf Chatdaten
        confidences = estimator.predict_proba(messages_data)

        # Sortieren mit Entropy und zurückgeben des höchsten Wertes
        indexed_confidences = [(i, e) for i, e in enumerate(confidences)]
        sorted_confidences = sorted(indexed_confidences, key=lambda value: scipy.stats.entropy(value[1]))
        return messages[sorted_confidences[-1][0]]

    async def generate_vocabulary_data(self, include_unlabeled=False):
        '''
        Erstellt eine Liste aller Chatnachrichten der Recordings des Klassifikators
        '''
        channel_filter = await self.create_channel_filter()
        id_filter = self.classifier.test_set
        label_filter = {} if include_unlabeled else {'label': {'$exists': True}}
        complete_filter = {**{'$or': channel_filter, '_id': {'$nin': id_filter}}, **label_filter}
        return Message.find(complete_filter)

    async def generate_review_data(self, include_unlabeled=False):
        """
        Erstellt eine Liste aller Chatnachrichten der Recordings des Klassifikators die nicht bereits im Test- oder
        Trainingsset enthalten sind.
        """
        channel_filter = await self.create_channel_filter()
        id_filter = self.classifier.test_set + self.classifier.train_set + self.classifier.unlabeled_train_set
        label_filter = {} if include_unlabeled else {'label': {'$exists': True}}
        complete_filter = {**{'$or': channel_filter, '_id': {'$nin': id_filter}}, **label_filter}
        review_cursor = Message.find(complete_filter).limit(10000)
        review_count = await review_cursor.count()
        log.info('Review Data loaded: {} documents'.format(review_count))
        return await review_cursor.to_list(None)

    async def generate_test_data(self, test_size=0.5, limit=None):
        """
        Erstellt eine Teilmenge der zur Verfügung stehenden Testmenge aus den konfigurierten Recordings des Klassifikators.
        """
        channel_filter = await self.create_channel_filter()

        # calc max count
        all_count = await Message.find({'$or': channel_filter}).count()
        max_count = None
        if test_size and limit:
            max_count = min(all_count, limit) * test_size
        elif test_size and not limit:
            max_count = all_count * test_size
        elif not test_size and limit:
            max_count = min(all_count, limit)

        positive = await Message.find_random({'$or': channel_filter, 'label': 3}, amount=max_count)
        negative = await Message.find_random({'$or': channel_filter, 'label': 2}, amount=max_count)
        neutral = await Message.find_random({'$or': channel_filter, 'label': 1}, amount=max_count)
        data = [message.id for message in positive + negative + neutral]
        return data

    async def generate_fixed_evaluation_data(self, limit=None):
        """
        Erstellt eine Testmenge aus den konfigurierten Recordings des Klassifikators.
        """
        channel_filter = await self.create_channel_filter()
        positive_cursor = Message.find({'$or': channel_filter, 'label': 3})
        negative_cursor = Message.find({'$or': channel_filter, 'label': 2})
        neutral_cursor = Message.find({'$or': channel_filter, 'label': 1})

        positives = await positive_cursor.to_list(limit)
        negatives = await negative_cursor.to_list(limit)
        neutrals = await neutral_cursor.to_list(limit)

        shuffled = neutrals + negatives + positives
        shuffle(shuffled)

        data = [message.content for message in shuffled]
        target = [message.label - 1 for message in shuffled]
        return data, target

    async def create_channel_filter(self):
        """
        Erstellt einen Filter für Chatnachrichten anhand aller Recordings die für den gesetzten Klassifikator
        ausgewählt wurden
        """
        future = Recording.find({'_id': {'$in': self.classifier.training_sets}})
        recordings = await future.to_list(None)
        return [
            {'channel_id': str(recording.channel_id), 'created': {'$gte': recording.started, '$lt': recording.stopped}}
            for recording in recordings]

    async def load_test_data(self):
        message_cursor = Message.find({'_id': {'$in': self.classifier.test_set}})
        return await DataSource.load_data(message_cursor)

    async def load_train_data(self):
        message_cursor = Message.find({'_id': {'$in': self.classifier.train_set}})
        return await DataSource.load_data(message_cursor)

    @staticmethod
    async def load_data(data_cursor):
        data = []
        target = []
        async for message in data_cursor:
            data.append(message.content)
            target.append(message.label - 1)

        return data, target
