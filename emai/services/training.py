from emai.persistence import Message, Bag, Recording
import asyncio
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
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


class ClassifierType(Enum):
    NaiveBayes = 1
    SupportVectorMachine = 2
    LogisticRegression = 3


class TrainingService(object):

    @staticmethod
    async def update_classifier(classifier, new_data):
        training_sets = new_data['training_sets']
        if training_sets:
            classifier.training_sets = [ObjectId(r_id) for r_id in training_sets]

        type = new_data['type']
        if type and ClassifierType(type):
            classifier.type = type

        settings = new_data['settings']
        if settings:
            classifier.settings = settings

        await classifier.commit()

    @staticmethod
    async def train(classifier):
        if not classifier.settings:
            raise ValueError('Classifier Settings must be set')
        if 'ngram_range' not in classifier.settings:
            raise ValueError('Classifier Settings ngram_range must be set')
        if 'stop_words' not in classifier.settings:
            raise ValueError('Classifier Settings stop_words must be set')
        if 'idf' not in classifier.settings:
            raise ValueError('Classifier Settings idf must be set')

        # generate sample data
        data, target = await TrainingService.generate_train_data(classifier)
        train_data, test_data, train_target, test_target = cross_validation.train_test_split(data, target, test_size=0.5, random_state=17)

        # configure pipeline
        ngram_range = (1, classifier.settings['ngram_range'])
        stop_words = 'english' if classifier.settings['stop_words'] else None
        count_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        tfidf_tranformer = TfidfTransformer(use_idf=classifier.settings['idf'])
        estimator = TrainingService.get_estimator(classifier)
        pipeline = Pipeline([('vect', count_vectorizer), ('tfidf', tfidf_tranformer), ('cls', estimator)])

        # train estimators
        pipeline.fit(train_data, train_target)

        # test estimators
        test_result = TrainingService.test_estimator(pipeline, test_data, test_target)
        print(test_result)
        #s = pickle.dumps(pipeline)
        #clf2 = pickle.loads(s)
        #test_result2 = TrainingService.test_estimator(clf2, test_data, test_target)


        await TrainingService.review(classifier ,pipeline)

    @staticmethod
    async def review(classifier, estimator):
        messages = await TrainingService.generate_review_data(classifier)
        messages_data = [message.content for message in messages]

        confidences = np.abs(estimator.decision_function(messages_data))
        average_confidences = np.average(confidences, axis=1)
        sorted_confidences = np.argsort(average_confidences)
        high_confidences = sorted_confidences[-5:]
        high_messages = [messages[message_id] for message_id in high_confidences.tolist()]
        low_confidences = sorted_confidences[0:5]
        low_messages = [messages[message_id] for message_id in low_confidences.tolist()]
        return high_messages, low_messages


    @staticmethod
    def get_estimator(classifier):
        classifier_type = ClassifierType(classifier.type)
        if classifier_type == ClassifierType.LogisticRegression:
            return LogisticRegression(random_state=42)
        elif classifier_type == ClassifierType.SupportVectorMachine:
            return SGDClassifier(random_state=10)
        elif classifier_type == ClassifierType.NaiveBayes:
            return MultinomialNB()

    @staticmethod
    def test_estimator(estimator, data, target):
        predicted = estimator.predict(data)
        precision, recall, fscore, support = precision_recall_fscore_support(target, predicted)
        return {
            'neutral': {
                'precision': precision[0].item(),
                'recall': recall[0].item(),
                'fscore': fscore[0].item(),
                'support': support[0].item()
            },
            'negative': {
                'precision': precision[1].item(),
                'recall': recall[1].item(),
                'fscore': fscore[1].item(),
                'support': support[1].item()
            },
            'positive': {
                'precision': precision[2].item(),
                'recall': recall[2].item(),
                'fscore': fscore[2].item(),
                'support': support[2].item()
            }
        }

        # # Logistic Regression
        # lr_classifier = LogisticRegression()
        # lr_classifier.fit(train_data_tfidf, train_target)
        # predicted = lr_classifier.predict(sample_data_tfidf)
        # print(metrics.classification_report(test_target, predicted, target_names=['negative', 'positive']))
        #
        #
        # # parameter tuning
        # parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
        #               'vect__stop_words': ['english', None],
        #               'tfidf__use_idf': (True, False),
        #               'clf__alpha': (1e-2, 1e-3)
        #               }
        #
        # sgd_pipeline = Pipeline([('vect', CountVectorizer()),
        #             ('tfidf', TfidfTransformer()),
        #                          ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
        # gs_clf = GridSearchCV(sgd_pipeline, parameters, n_jobs=-1)
        # gs_clf.fit(train_data, train_target)
        #
        # best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        # for param_name in sorted(parameters.keys()):
        #     print("%s: %r" % (param_name, best_parameters[param_name]))
        #
        # predicted = gs_clf.predict(test_data)
        # print(metrics.classification_report(test_target, predicted, target_names=['negative', 'positive']))


    @staticmethod
    async def generate_train_data(classifier):
        channel_filter = await TrainingService.create_channel_filter(classifier)
        positive_cursor = Message.find({'$or': channel_filter, 'label': 3})
        negative_cursor = Message.find({'$or': channel_filter, 'label': 2})
        neutral_cursor = Message.find({'$or': channel_filter, 'label': 1})

        positive_count = await positive_cursor.count()
        negative_count = await negative_cursor.count()
        neutral_count = await neutral_cursor.count()
        total_count = positive_count + negative_count + neutral_count
        log.info('Train Data loaded: Positive: ~{}% Negative: ~{}% Neutral: ~{}%'.format(int(100/total_count * positive_count), int(100/total_count * negative_count), int(100/total_count * neutral_count)))

        positives = [message.content for message in await positive_cursor.to_list(None)]
        negatives = [message.content for message in await negative_cursor.to_list(None)]
        neutrals = [message.content for message in await neutral_cursor.to_list(None)]

        data = neutrals + negatives + positives
        target = [0] * neutral_count + [1] * negative_count + [2] * positive_count
        return data, target

    @staticmethod
    async def generate_review_data(classifier):
        channel_filter = await TrainingService.create_channel_filter(classifier)
        unlabeled_cursor = Message.find({'$or': channel_filter, 'label': {'$exists': False}}).limit(10000)
        unlabeled_count = await unlabeled_cursor.count()
        log.info('Review Data loaded: {} documents'.format(unlabeled_count))
        return await unlabeled_cursor.to_list(None)


    @staticmethod
    async def create_channel_filter(classifier):
        recordings = await Recording.find({'id': {'$in': classifier.training_sets}}).to_list(None)
        return [{'channel_id': str(recording.channel_id), 'created': {'$gte': recording.started, '$lt': recording.stopped}} for recording in recordings]





