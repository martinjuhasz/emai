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
from emai.persistence import Message, Recording, Performance
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
    def transformation(text):
        return text

    @staticmethod
    def max_feature_count(count_vectorizer, min_occurences, train_data):
        vocabulary_matrix = count_vectorizer.fit_transform(train_data)
        vocabulary_count = np.asarray(vocabulary_matrix.sum(axis=0)).ravel()
        excluded = np.sum(i > min_occurences for i in vocabulary_count)
        return len(vocabulary_count) - excluded


class DataSource(object):
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
        amount = math.ceil(amount / 3)
        channel_filter = await self.create_channel_filter()
        positive = await Message.get_random(channel_filter, 3, amount)
        negative = await Message.get_random(channel_filter, 2, amount)
        neutral = await Message.get_random(channel_filter, 1, amount)
        ids = [message.id for message in positive + negative + neutral]
        return ids

    async def get_max_train_count(self):
        channel_filter = await self.create_channel_filter()
        message_filter = {**{'$or': channel_filter}, **{'label': {'$exists': True}}}
        return await Message.find(message_filter).count()

    async def random_message(self, interactive=False):
        channel_filter = await self.create_channel_filter()
        label = randint(1, 3) if not interactive else None
        messages = await Message.get_random(channel_filter, label)
        return messages[0]

    async def generate_informative_messages(self, interactive=False):
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

        # search for messages that contain 1/3 top vocabulary
        top_searches = [re.compile(token, re.I) for token, token_count in vocabulary[:10]]
        channel_filter = await self.create_channel_filter()
        label_filter = {} if interactive else {'label': {'$exists': True}}
        message_cursor = Message.find({**{'$or': channel_filter, 'content': {'$in': top_searches}}, **label_filter})
        self.informative_stack = message_cursor

    async def most_informative(self, interactive=False):
        if not self.informative_stack:
            await self.generate_informative_messages(interactive=interactive)

        if await self.informative_stack.fetch_next:
            msg = self.informative_stack.next_object()
            return msg
        else:
            return None

    async def least_confident_message(self, estimator, interactive=False):
        messages = await self.generate_review_data(include_unlabeled=interactive)
        messages_data = [message.content for message in messages]

        classifier_type = ClassifierType(self.classifier.type)
        if classifier_type == ClassifierType.NaiveBayes:
            confidences = np.abs(estimator.decision_function(messages_data))
        else:
            confidences = estimator.predict_proba(messages_data)
        indexed_confidences = [(i, e) for i, e in enumerate(confidences)]
        # sorted_confidences = sorted(indexed_confidences, key=lambda value: DataSource.entropy(value[1]))
        sorted_confidences = sorted(indexed_confidences, key=lambda value: scipy.stats.entropy(value[1]))
        return messages[sorted_confidences[-1][0]]

        # confidences = np.abs(estimator.decision_function(messages_data))
        # average_confidences = np.average(confidences, axis=1)
        # sorted_confidences = np.argsort(average_confidences)
        # return messages[sorted_confidences[0]]

        # confidences = np.abs(estimator.decision_function(messages_data))
        # indexed_confidences = [(i, e) for i, e in enumerate(confidences)]
        # sorted_confidences = sorted(indexed_confidences, key=lambda value: DataSource.entropy(value[1]))
        # return messages[sorted_confidences[-1][0]]

        # confidences = estimator.predict_proba(messages_data)
        # sorted_confidences = sorted(confidences, key=lambda value: np.abs(np.average(value) - 0.5))
        # return messages[sorted_confidences[-1]]

    async def generate_vocabulary_data(self, include_unlabeled=False):
        channel_filter = await self.create_channel_filter()
        id_filter = self.classifier.test_set
        label_filter = {} if include_unlabeled else {'label': {'$exists': True}}
        complete_filter = {**{'$or': channel_filter, '_id': {'$nin': id_filter}}, **label_filter}
        return Message.find(complete_filter)

    async def generate_review_data(self, include_unlabeled=False):
        channel_filter = await self.create_channel_filter()
        id_filter = self.classifier.test_set + self.classifier.train_set + self.classifier.unlabeled_train_set
        label_filter = {} if include_unlabeled else {'label': {'$exists': True}}
        complete_filter = {**{'$or': channel_filter, '_id': {'$nin': id_filter}}, **label_filter}
        review_cursor = Message.find(complete_filter).limit(10000)
        review_count = await review_cursor.count()
        log.info('Review Data loaded: {} documents'.format(review_count))
        return await review_cursor.to_list(None)

    async def generate_test_data(self, test_size=0.5, limit=None):
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

        positive_cursor = Message.find({'$or': channel_filter, 'label': 3}).limit(max_count)
        negative_cursor = Message.find({'$or': channel_filter, 'label': 2}).limit(max_count)
        neutral_cursor = Message.find({'$or': channel_filter, 'label': 1}).limit(max_count)

        positive_count = await positive_cursor.count()
        negative_count = await negative_cursor.count()
        neutral_count = await neutral_cursor.count()
        total_count = positive_count + negative_count + neutral_count
        log.info('Test Data loaded: Total/Labeled: {}/{} Positive: ~{}% Negative: ~{}% Neutral: ~{}%'.format(all_count,
                                                                                                             total_count,
                                                                                                             int(
                                                                                                                 100 / total_count * positive_count),
                                                                                                             int(
                                                                                                                 100 / total_count * negative_count),
                                                                                                             int(
                                                                                                                 100 / total_count * neutral_count)))

        positives = [message.id for message in await positive_cursor.to_list(None)]
        negatives = [message.id for message in await negative_cursor.to_list(None)]
        neutrals = [message.id for message in await neutral_cursor.to_list(None)]

        data = neutrals + negatives + positives
        target = [0] * neutral_count + [1] * negative_count + [2] * positive_count
        train_data, test_data, train_target, test_target = cross_validation.train_test_split(data, target,
                                                                                             test_size=test_size,
                                                                                             random_state=17)
        return test_data

    async def generate_fixed_evaluation_data(self, limit=None):
        channel_filter = await self.create_channel_filter()
        positive_cursor = Message.find({'$or': channel_filter, 'label': 3})
        negative_cursor = Message.find({'$or': channel_filter, 'label': 2})
        neutral_cursor = Message.find({'$or': channel_filter, 'label': 1})

        positive_count = await positive_cursor.count()
        negative_count = await negative_cursor.count()
        neutral_count = await neutral_cursor.count()

        positives = await positive_cursor.to_list(limit)
        negatives = await negative_cursor.to_list(limit)
        neutrals = await neutral_cursor.to_list(limit)

        shuffled = neutrals + negatives + positives
        shuffle(shuffled)

        data = [message.content for message in shuffled]
        target = [message.label - 1 for message in shuffled]
        return data, target

    async def create_channel_filter(self):
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


class Trainer(object):
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

    async def ensure_train_set(self):
        if self.classifier.has_train_set():
            return
        self.classifier.train_set = await self.datasource.get_sample_train_set()

    async def train(self):
        self.ensure_configured()
        await self.ensure_train_set()

        # generate classifier data
        train_data, train_target = await self.datasource.load_train_data()

        # configure pipeline
        ngram_range = (1, self.classifier.settings['ngram_range'])
        stop_words = PreProcessing.stop_words if self.classifier.settings['stop_words'] else None
        count_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        tfidf_tranformer = TfidfTransformer(use_idf=self.classifier.settings['idf'])
        estimator = self.choose_estimator()
        pipeline = Pipeline([('vect', count_vectorizer), ('tfidf', tfidf_tranformer), ('cls', estimator)])

        # train estimators
        pipeline.fit(train_data, train_target)

        self.estimator = pipeline

    def choose_estimator(self):
        classifier_type = ClassifierType(self.classifier.type)
        if classifier_type == ClassifierType.LogisticRegression:
            return LogisticRegression(random_state=42, C=2)
        elif classifier_type == ClassifierType.SupportVectorMachine:
            return SVC(random_state=10, C=2, gamma=0.5)
        elif classifier_type == ClassifierType.NaiveBayes:
            return MultinomialNB(alpha=0.5)

    async def save(self):
        classifier_state = pickle.dumps(self.estimator)
        self.classifier.state = classifier_state
        await self.classifier.commit()

    def load(self):
        if self.classifier.state and not self.estimator:
            self.estimator = pickle.loads(self.classifier.state)

    def is_waiting_for_mentoring(self):
        # cannot learn since next messages need to be labeled first
        if self.classifier.unlabeled_train_set and len(self.classifier.unlabeled_train_set) >= 0:
            return True
        return False

    def add_for_mentoring(self, message_id):
        self.classifier.unlabeled_train_set.append(message_id)

    async def check_for_mentoring(self):
        newly_labeled = []
        for message_id in self.classifier.unlabeled_train_set:
            message = await Message.find_one({'id': message_id})
            if message.label and message.label > 0:
                newly_labeled.append(message_id)
                self.classifier.train_set.append(message_id)
        new_unlabeled_set = [mid for mid in self.classifier.unlabeled_train_set if mid not in newly_labeled]
        self.classifier.unlabeled_train_set = new_unlabeled_set

    async def train_until(self, train_count, test_iterative=False):
        if not train_count:
            train_count = await self.datasource.get_max_train_count()

        current_train_count = len(self.classifier.train_set)
        if train_count <= current_train_count:
            return

        add_train_set = await self.datasource.get_sample_train_set(amount=train_count - current_train_count)
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
        # check first if some mentoring was done or if mentoring is needed
        await self.check_for_mentoring()
        if self.is_waiting_for_mentoring():
            return

        await self.train()
        if test:
            await self.test()

        # learn continuous while prelabeled data exists
        randomize_counter = 1
        while not self.is_waiting_for_mentoring():
            if max_learn_count and len(self.classifier.train_set) > max_learn_count:
                break

            # check if next message needs mentoring
            if not randomize or (randomize and randomize_step and randomize_counter % randomize_step == 0):
                if learn_type == LearnType.LeastConfident:
                    next_message = await self.datasource.least_confident_message(self.estimator,
                                                                                 interactive=interactive)
                else:
                    next_message = await self.datasource.most_informative(interactive=interactive)
            else:
                next_message = await self.datasource.random_message(interactive=interactive)

            # break if no messages are left to train
            if not next_message:
                break

            if not next_message.label:
                self.add_for_mentoring(next_message.id)
                continue

            # update classifier with new message
            randomize_counter += 1
            self.classifier.train_set.append(next_message.id)
            await self.train()
            if test:
                await self.test()

        if save:
            await self.save()

    async def messages_for_mentoring(self):
        if not self.is_waiting_for_mentoring():
            return None

        cursor = Message.find({'_id': {'$in': self.classifier.unlabeled_train_set}})
        return await cursor.to_list(None)

    async def test(self):
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
        if not self.test_data or not self.test_target:
            self.test_data, self.test_target = await self.datasource.load_test_data()
        train_data, train_target = await self.datasource.load_train_data()

        scores = [self.estimator.score(train_data, train_target),
                  self.estimator.score(self.test_data, self.test_target)]
        return scores


class TrainingService(object):
    @staticmethod
    async def update_classifier(classifier, new_data):
        if 'training_sets' in new_data:
            classifier.training_sets = [ObjectId(r_id) for r_id in new_data['training_sets']]

        if 'type' in new_data and ClassifierType(new_data['type']):
            classifier.type = new_data['type']

        if 'settings' in new_data:
            if any(setting in new_data['settings'] for setting in ('ngram_range', 'stop_words', 'idf')):
                classifier.settings = new_data['settings']

        # generate new test set
        data_source = DataSource(classifier)
        classifier.test_set = await data_source.generate_test_data()

        await classifier.commit()
        await classifier.reset()

    @staticmethod
    async def train(classifier, train_count=None):
        trainer = Trainer(classifier)
        trainer.ensure_configured()

        await trainer.train_until(train_count, test_iterative=True)

    @staticmethod
    async def learn(classifier):
        trainer = Trainer(classifier)
        trainer.ensure_configured()

        await trainer.learn()

        return await trainer.messages_for_mentoring()
