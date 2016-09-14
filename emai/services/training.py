from emai.persistence import Message, Recording, Performance, PerformanceResult
import asyncio
from time import perf_counter as pc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics
from bson import ObjectId
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from emai.utils import log, config
from sklearn.metrics import precision_recall_fscore_support
from enum import Enum
import pickle
from datetime import datetime
from random import shuffle


class ClassifierType(Enum):
    NaiveBayes = 1
    SupportVectorMachine = 2
    LogisticRegression = 3


class PreProcessing(object):

    stop_words = ['aber', 'als', 'am', 'an', 'auch', 'auf', 'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'dadurch', 'daher', 'darum', 'das', 'daß', 'dass', 'dein', 'deine', 'dem', 'den', 'der', 'des', 'dessen', 'deshalb', 'die', 'dies', 'dieser', 'dieses', 'doch', 'dort', 'du', 'durch', 'ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'er', 'es', 'euer', 'eure', 'für', 'hatte', 'hatten', 'hattest', 'hattet', 'hier', 'hinter', 'ich', 'ihr', 'ihre', 'im', 'in', 'ist', 'ja', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jener', 'jenes', 'jetzt', 'kann', 'kannst', 'können', 'könnt', 'machen', 'mein', 'meine', 'mit', 'muß', 'mußt', 'musst', 'müssen', 'müßt', 'nach', 'nachdem', 'nein', 'nicht', 'nun', 'oder', 'seid', 'sein', 'seine', 'sich', 'sie', 'sind', 'soll', 'sollen', 'sollst', 'sollt', 'sonst', 'soweit', 'sowie', 'und', 'unser', 'unsere', 'unter', 'vom', 'von', 'vor', 'wann', 'warum', 'was', 'weiter', 'weitere', 'wenn', 'wer', 'werde', 'werden', 'werdet', 'weshalb', 'wie', 'wieder', 'wieso', 'wir', 'wird', 'wirst', 'wo', 'woher', 'wohin', 'zu', 'zum', 'zur', 'über', "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

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

    async def get_sample_train_set(self, amount=3):
        channel_filter = await self.create_channel_filter()
        positive = await Message.get_random(channel_filter, 3, amount)
        negative = await Message.get_random(channel_filter, 2, amount)
        neutral = await Message.get_random(channel_filter, 1, amount)
        ids = [message.id for message in positive + negative + neutral]
        return ids

    async def least_confident_message(self, estimator):
        messages = await self.generate_review_data()
        messages_data = [message.content for message in messages]

        confidences = np.abs(estimator.decision_function(messages_data))
        average_confidences = np.average(confidences, axis=1)
        sorted_confidences = np.argsort(average_confidences)
        return messages[sorted_confidences[0]]

        #confidences = estimator.predict_proba(messages_data)
        #sorted_confidences = sorted(confidences, key=lambda value: np.abs(np.average(value) - 0.5))
        #return messages[sorted_confidences[-1]]


    async def  generate_review_data(self):
        channel_filter = await self.create_channel_filter()
        id_filter = self.classifier.test_set + self.classifier.train_set + self.classifier.unlabeled_train_set
        review_cursor = Message.find({'$or': channel_filter, '_id': {'$nin': id_filter}}).limit(10000)
        review_count = await review_cursor.count()
        log.info('Review Data loaded: {} documents'.format(review_count))
        return await review_cursor.to_list(None)

    async def generate_test_data(self, test_size=0.5):
        channel_filter = await self.create_channel_filter()
        all_cursor = Message.find({'$or': channel_filter})
        positive_cursor = Message.find({'$or': channel_filter, 'label': 3})
        negative_cursor = Message.find({'$or': channel_filter, 'label': 2})
        neutral_cursor = Message.find({'$or': channel_filter, 'label': 1})

        all_count = await all_cursor.count()
        positive_count = await positive_cursor.count()
        negative_count = await negative_cursor.count()
        neutral_count = await neutral_cursor.count()
        total_count = positive_count + negative_count + neutral_count
        log.info('Test Data loaded: Total/Labeled: {}/{} Positive: ~{}% Negative: ~{}% Neutral: ~{}%'.format(all_count, total_count,
            int(100 / total_count * positive_count), int(100 / total_count * negative_count),
            int(100 / total_count * neutral_count)))

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
        self.test_set = None
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
        count_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words, preprocessor=PreProcessing.transformation)
        tfidf_tranformer = TfidfTransformer(use_idf=self.classifier.settings['idf'])
        estimator = self.choose_estimator()
        pipeline = Pipeline([('vect', count_vectorizer), ('tfidf', tfidf_tranformer), ('cls', estimator)])

        # train estimators
        pipeline.fit(train_data, train_target)
        self.estimator = pipeline

    def choose_estimator(self):
        classifier_type = ClassifierType(self.classifier.type)
        if classifier_type == ClassifierType.LogisticRegression:
            return LogisticRegression(random_state=42)
        elif classifier_type == ClassifierType.SupportVectorMachine:
            return SVC(kernel='linear', random_state=10)
        elif classifier_type == ClassifierType.NaiveBayes:
            return MultinomialNB()

    async def save(self):
        classifier_state = pickle.dumps(self.estimator)
        self.classifier.state = classifier_state
        await self.classifier.commit()

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

    async def learn(self):
        # check first if some mentoring was done or if mentoring is needed
        await self.check_for_mentoring()
        if self.is_waiting_for_mentoring():
            return
        await self.train()
        await self.test()

        # learn continuous while prelabeled data exists
        while not self.is_waiting_for_mentoring():

            # check if next message needs mentoring
            next_message = await self.datasource.least_confident_message(self.estimator)
            if not next_message.label:
                self.add_for_mentoring(next_message.id)
                continue

            # update classifier with new message
            self.classifier.train_set.append(next_message.id)
            await self.train()
            await self.test()

        await self.save()

    async def messages_for_mentoring(self):
        if not self.is_waiting_for_mentoring():
            return None

        cursor = Message.find({'_id': {'$in': self.classifier.unlabeled_train_set}})
        return await cursor.to_list(None)

    async def test(self):
        if not self.test_set:
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
    async def learn(classifier):
        trainer = Trainer(classifier)
        trainer.ensure_configured()

        await trainer.learn()

        return await trainer.messages_for_mentoring()




