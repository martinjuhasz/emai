from emai.persistence import Message, Bag
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
from emai.utils import log
from sklearn.metrics import precision_recall_fscore_support

class TrainingService(object):

    def __init__(self):
        pass

    @staticmethod
    async def test(recording_id):
        # search trained bags
        # bags = Bag.get_training_messages(recording_id)
        # data = []
        # target = []
        # for document in (await bags.to_list(None)):
        #     data.append(document['content'])
        #     target.append(document['label'] - 2)

        data, target = await TrainingService.generate_sample_data(recording_id)

        # create training data
        train_data, test_data, train_target, test_target = cross_validation.train_test_split(data, target, test_size=0.5, random_state=17)

        # train classifiers
        nb_classifier_result = ClassifierResult()
        nb_classifier_result.train(MultinomialNB(), train_data, train_target)
        svm_classifier_result = ClassifierResult()
        svm_classifier_result.train(SGDClassifier(random_state=42), train_data, train_target, {
            'clf__alpha': (1e-2, 1e-3),
            'clf__loss': ['hinge', 'log', 'modified_huber']
        })
        logreg_classifier_result = ClassifierResult()
        logreg_classifier_result.train(LogisticRegression(), train_data, train_target)
        log.info('Created NaiveBayes Classifier: {}'.format(nb_classifier_result.settings))
        log.info('Created SupportVectorMachine Classifier: {}'.format(svm_classifier_result.settings))
        log.info('Created LogisticRegression Classifier: {}'.format(logreg_classifier_result.settings))

        # test classifiers
        nb_classifier_result.test(test_data, test_target)
        svm_classifier_result.test(test_data, test_target)
        logreg_classifier_result.test(test_data, test_target)
        log.info('Tested NaiveBayes Classifier: {}'.format(nb_classifier_result.results))
        log.info('Tested SupportVectorMachine Classifier: {}'.format(svm_classifier_result.results))
        log.info('Tested LogisticRegression Classifier: {}'.format(logreg_classifier_result.results))

        # probe classifiers
        unsampled_bags = Bag.get_training_messages(recording_id, label_eq={'$exists': False}, limit=300)
        unsampled_data = [document['content'] for document in await unsampled_bags.to_list(None)]
        nb_classifier_result.probe(unsampled_data)
        #svm_classifier_result.probe(unsampled_data)
        logreg_classifier_result.probe(unsampled_data)
        log.info('Probed NaiveBayes Classifier')
        #log.info('Probed SupportVectorMachine Classifier')
        log.info('Probed LogisticRegression Classifier')

        return {
            'nb': nb_classifier_result,
            'svm': svm_classifier_result,
            'logreg': logreg_classifier_result
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
    async def generate_sample_data(recording_id):
        positive_count = len(await Bag.get_training_messages(recording_id, label_eq=3).to_list(None))
        negative_count = len(await Bag.get_training_messages(recording_id, label_eq=2).to_list(None))
        samples = min(positive_count, negative_count)
        positives = await Bag.get_training_messages(recording_id, label_eq=3, samples=samples).to_list(None)
        negatives = await Bag.get_training_messages(recording_id, label_eq=2, samples=samples).to_list(None)
        data = []
        target = []
        for sample_id in range(0, samples):
            data.append(positives[sample_id]['content'])
            target.append(1)
            data.append(negatives[sample_id]['content'])
            target.append(0)

        return data, target



class ClassifierResult(object):
    classifier = None
    settings = {}
    results = None
    vocabulary = None
    probes = {}

    def train(self, classifier, data, target, parameters={}):
        pipeline = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', classifier)])
        combined_parameters = {**{
            'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'vect__stop_words': ['english', None],
            'tfidf__use_idf': (True, False)
        }, **parameters}
        grid_search = GridSearchCV(pipeline, combined_parameters, n_jobs=-1)
        grid_search.fit(data, target)
        best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])

        # run count vectorizor again to vizualize dictionary
        #sample_count_vectorizer = CountVectorizer(ngram_range=best_parameters['vect__ngram_range'], stop_words=best_parameters['vect__stop_words'])
        #sample_count_vectorizer.fit_transform(data)
        #self.vocabulary = sample_count_vectorizer.vocabulary_

        self.classifier = grid_search
        self.settings = best_parameters

    def test(self, data, target):
        predicted = self.classifier.predict(data)
        precision, recall, fscore, support = precision_recall_fscore_support(target, predicted)
        self.results = {
            'negative': {
                'precision': precision[0].item(),
                'recall': recall[0].item(),
                'fscore': fscore[0].item(),
                'support': support[0].item()
            },
            'positive': {
                'precision': precision[1].item(),
                'recall': recall[1].item(),
                'fscore': fscore[1].item(),
                'support': support[1].item()
            }
        }

    def probe(self, data):
        if 'clf__loss' in self.settings and 'hinge' == self.settings['clf__loss']:
            return
        proba = self.classifier.predict_proba(data)
        top_positive = proba[:, 1].argpartition(-100)[-100:]
        top_negative = proba[:, 0].argpartition(-100)[-100:]
        self.probes['positive'] = [(data[top_idx], proba[top_idx][1].item()) for top_idx in top_positive]
        self.probes['negative'] = [(data[top_idx], proba[top_idx][0].item()) for top_idx in top_negative]

    def get_results(self):
        return {
            'settings': self.settings,
            'results': self.results,
            'vocabulary': self.vocabulary,
            'probes': self.probes
        }

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    loop.run_until_complete(TrainingService.test(ObjectId('578fbf877b958024092b8e63')))
