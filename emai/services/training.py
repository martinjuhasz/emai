from emai.persistence import Message, Recording, Performance, PerformanceResult
import asyncio
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
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

        await classifier.commit()
        await classifier.reset()


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
        count_vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words, preprocessor=PreProcessing.transformation)
        feature_count = PreProcessing.max_feature_count(count_vectorizer, 2, train_data)
        count_vectorizer.max_features = feature_count

        tfidf_tranformer = TfidfTransformer(use_idf=classifier.settings['idf'])
        estimator = TrainingService.get_estimator(classifier)
        pipeline = Pipeline([('vect', count_vectorizer), ('tfidf', tfidf_tranformer), ('cls', estimator)])

        # train estimators
        pipeline.fit(train_data, train_target)

        # test
        #vocabulary_matrix = pipeline.steps[0][1].transform(train_data)
        #vocabulary = sorted(list(zip(pipeline.steps[0][1].get_feature_names(), np.asarray(vocabulary_matrix.sum(axis=0)).ravel().tolist())), key=lambda item: item[1], reverse=True)

        # test estimators
        await TrainingService.test_classifier(classifier, pipeline, test_data, test_target)


        # update classifier
        classifier_state = pickle.dumps(pipeline)
        classifier.state = classifier_state
        await classifier.commit()

        return classifier

    @staticmethod
    async def review(classifier):
        estimator = pickle.loads(classifier.state)
        messages = await TrainingService.generate_review_data(classifier)
        messages_data = [message.content for message in messages]

        confidences = np.abs(estimator.decision_function(messages_data))
        average_confidences = np.average(confidences, axis=1)
        sorted_confidences = np.argsort(average_confidences)
        high_confidences = sorted_confidences[-5:]
        high_messages = [messages[message_id] for message_id in high_confidences.tolist()]
        low_confidences = sorted_confidences[0:25]
        low_messages = [messages[message_id] for message_id in low_confidences.tolist()]
        return_messages = high_messages + low_messages
        np.random.shuffle(return_messages)
        return return_messages


    @staticmethod
    def get_estimator(classifier):
        classifier_type = ClassifierType(classifier.type)
        if classifier_type == ClassifierType.LogisticRegression:
            return LogisticRegression(random_state=42)
        elif classifier_type == ClassifierType.SupportVectorMachine:
            return LinearSVC(random_state=10)
        elif classifier_type == ClassifierType.NaiveBayes:
            return MultinomialNB()

    @staticmethod
    async def test_classifier(classifier, estimator, data, target):
        predicted = estimator.predict(data)
        precision, recall, fscore, support = precision_recall_fscore_support(target, predicted)

        time = datetime.utcnow()
        if not classifier.performance:
            classifier.performance = Performance.create_empty_performance()
        classifier.performance._modified = True  # TODO: is this a bug? shouldn't have to do that: report

        classifier.performance.neutral.time.append(time)
        classifier.performance.neutral.precision.append(precision[0].item())
        classifier.performance.neutral.recall.append(recall[0].item())
        classifier.performance.neutral.fscore.append(fscore[0].item())
        classifier.performance.neutral.support.append(support[0].item())

        classifier.performance.negative.time.append(time)
        classifier.performance.negative.precision.append(precision[1].item())
        classifier.performance.negative.recall.append(recall[1].item())
        classifier.performance.negative.fscore.append(fscore[1].item())
        classifier.performance.negative.support.append(support[1].item())

        classifier.performance.positive.time.append(time)
        classifier.performance.positive.precision.append(precision[2].item())
        classifier.performance.positive.recall.append(recall[2].item())
        classifier.performance.positive.fscore.append(fscore[2].item())
        classifier.performance.positive.support.append(support[2].item())

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





