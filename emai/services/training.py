from emai.persistence import Message, Bag
import pprint
import asyncio
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from bson import ObjectId
import numpy as np

class TrainingService(object):

    def __init__(self):
        pass


    async def test(self):
        pass

        bags = Bag.get_training_bags(ObjectId('578fbf877b958024092b8e63'), 1)
        data = []
        target = []
        for document in (await bags.to_list(None)):
            data.append(' \n '.join([message['content'] for message in document['messages']]))
            target.append(document['label'] - 2)

        train_data, test_data, train_target, test_target = cross_validation.train_test_split(data, target, test_size=0.3, random_state=17)

        # train
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(train_data)
        #print(X_train_counts.shape)
        #pprint.pprint(count_vect.vocabulary_)
        #pprint.pprint(count_vect.stop_words_)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        #pprint.pprint(X_train_tfidf.shape)

        #sample
        X_new_counts = count_vect.transform(test_data)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        #predict
        clf = MultinomialNB().fit(X_train_tfidf, train_target)
        predicted = clf.predict(X_new_tfidf)
        #pprint.pprint(mean)
        ##categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        #twenty_train = fetch_20newsgroups(subset='train', categories = categories, shuffle = True, random_state = 42)
        #pprint.pprint(twenty_train.data)

        print(metrics.classification_report(test_target, predicted, target_names=['negative', 'positive']))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    training = TrainingService()
    loop.run_until_complete(training.test())
