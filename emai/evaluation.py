import asyncio
from emai.persistence import Classifier
from emai.services.training import ClassifierType, DataSource, PreProcessing, Trainer, LearnType
from sklearn.learning_curve import learning_curve, _translate_train_sizes
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit
from bson import ObjectId
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy
import click
import pickle


# evaluation_recording = ObjectId('578fbf877b958024092b8e63')  # RBTV
evaluation_recording = ObjectId('5798ea0a7b95805f6e4dc1b2')  # Lassiz
# evaluation_recording = ObjectId('57e4e64c7b9580154ffa49b7')  # Burke

async def mentor_messages(messages):
    if not messages or len(messages) <= 0:
        return
    click.echo('Classify these messages:')
    click.secho('1) neutral', color='yellow')
    click.secho('2) negative', color='red')
    click.secho('3) positive', color='green')
    for message in messages:
        value = click.prompt(message.content, type=click.IntRange(1, 3))
        message.label = value
        await message.commit()


async def active_learning_curve(trainer, iterations, train_sizes, max_size):
    out = []
    train_sizes_abs = _translate_train_sizes(train_sizes, max_size)
    n_unique_ticks = train_sizes_abs.shape[0]
    for iteration in range(0, iterations):
        # setup test set for classifier
        await trainer.classifier.reset()
        trainer.reset()
        data_source = DataSource(trainer.classifier)
        trainer.classifier.test_set = await data_source.generate_test_data(limit=300, test_size=0.8)

        for train_size in train_sizes_abs:
            while len(trainer.classifier.train_set) < train_size:
                await trainer.learn(save=False, test=False, max_learn_count=train_size, randomize=True, randomize_step=False, interactive=False, learn_type=LearnType.LeastConfident)
                click.echo("Status: {}/{} iterations, {}/{} folds, {}/{} train size".format(iteration + 1, iterations, train_size, n_unique_ticks, len(trainer.classifier.train_set), train_size))
                messages = await trainer.messages_for_mentoring()
                await mentor_messages(messages)
            scores = await trainer.score()
            out.append(scores)

    out = numpy.array(out)
    n_cv_folds = out.shape[0] // n_unique_ticks
    out = out.reshape(n_cv_folds, n_unique_ticks, 2)
    out = numpy.asarray(out).transpose((2, 1, 0))

    return train_sizes_abs, out[0], out[1]


async def evaluate_active_learning():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.title = "Active Learning Evaluator"
    classifier.training_sets = [evaluation_recording]
    classifier.type = ClassifierType.LogisticRegression.value
    classifier.settings = {
        'ngram_range': 1,
        'stop_words': True,
        'idf': False
    }

    # load data
    ylim = [0.3, 1.0]
    #ylim = None


    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(7, 5)

    # Test Logistic Regression
    trainer = Trainer(classifier)
    train_sizes, train_scores, test_scores = await active_learning_curve(trainer, 20, numpy.linspace(0.05, 1., 10), 340)
    pickle.dump((train_sizes, train_scores, test_scores), open("al.dump", "wb"))
    plot_learning_curve(figure, [1, 1, 1], train_sizes, train_scores, test_scores, title="AL LeastConfident - LogisticRegression - C=2", ylim=ylim)

    figure.tight_layout()
    print('ended')

async def plot_last_active_learning():
    ylim = [0.3, 1.0]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(7, 5)

    train_sizes, train_scores, test_scores = pickle.load(open("al.dump", "rb"))
    plot_learning_curve(figure, [1, 1, 1], train_sizes, train_scores, test_scores, title="LogisticRegression - C=2", ylim=ylim)

    figure.tight_layout()
    print('ended')

async def evaluate_preprocessing_stopwords():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.45, 0.75]

    # configure pipelines
    stop_words = PreProcessing.stop_words
    pp_default = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]
    pp_stopwords = [('vect', CountVectorizer(stop_words=stop_words)), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 5)

    # Test Logistic Regression
    estimator = LogisticRegression(random_state=42)
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_stopwords = Pipeline(pp_stopwords + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 1], [pipeline_default, pipeline_stopwords], ["stopwords = false", "stopwords = true"], ["r", "g"], data, target, cv, title="Logistic Regression", ylim=ylim)

    # Test SVM Linear
    estimator = SVC(kernel='linear', random_state=42)
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_stopwords = Pipeline(pp_stopwords + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 2], [pipeline_default, pipeline_stopwords],
                       ["stopwords = false", "stopwords = true"], ["r", "g"], data, target, cv,
                       title="SVM linear", ylim=ylim)

    # Test Naive Bayes
    estimator = MultinomialNB()
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_stopwords = Pipeline(pp_stopwords + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 3], [pipeline_default, pipeline_stopwords],
                       ["stopwords = false", "stopwords = true"], ["r", "g"], data, target, cv,
                       title="Naive Bayes", ylim=ylim)


    figure.tight_layout()
    print('ended')


async def evaluate_preprocessing_ngram():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.40, 0.75]

    # configure pipelines
    pp_default = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]
    pp_ngram2 = [('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer())]
    pp_ngram3 = [('vect', CountVectorizer(ngram_range=(1, 3))), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 5)

    # Test Logistic Regression
    estimator = LogisticRegression(random_state=42)
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_ngram2 = Pipeline(pp_ngram2 + [('cls', estimator)])
    pipeline_ngram3 = Pipeline(pp_ngram3 + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 1], [pipeline_default, pipeline_ngram2, pipeline_ngram3], ["1 N-Gramm", "2 N-Gramm", "3 N-Gramm"], ["r", "g", "b"], data, target, cv, title="Logistic Regression", ylim=ylim)

    # Test SVM
    estimator = SVC(kernel='linear', random_state=42)
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_ngram2 = Pipeline(pp_ngram2 + [('cls', estimator)])
    pipeline_ngram3 = Pipeline(pp_ngram3 + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 2], [pipeline_default, pipeline_ngram2, pipeline_ngram3],
                       ["1 N-Gramm", "2 N-Gramm", "3 N-Gramm"], ["r", "g", "b"], data, target, cv,
                       title="SVM linear", ylim=ylim)

    # Test Naive Bayes
    estimator = MultinomialNB()
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_ngram2 = Pipeline(pp_ngram2 + [('cls', estimator)])
    pipeline_ngram3 = Pipeline(pp_ngram3 + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 3], [pipeline_default, pipeline_ngram2, pipeline_ngram3],
                       ["1 N-Gramm", "2 N-Gramm", "3 N-Gramm"], ["r", "g", "b"], data, target, cv,
                       title="Naive Bayes", ylim=ylim)

    figure.tight_layout()
    print('ended')

async def evaluate_preprocessing_idf():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.45, 0.75]

    # configure pipelines
    pp_default = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]
    pp_woidf = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer(use_idf=False))]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 5)

    # Test Logistic Regression
    estimator = LogisticRegression(random_state=42)
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_woidf = Pipeline(pp_woidf + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 1], [pipeline_default, pipeline_woidf], ["idf = true", "idf = false"], ["g", "r"], data, target, cv, title="Logistic Regression", ylim=ylim)

    # Test SVM Linear
    estimator = SVC(kernel='linear', random_state=42)
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_woidf = Pipeline(pp_woidf + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 2], [pipeline_default, pipeline_woidf], ["idf = true", "idf = false"], ["g", "r"],
                       data, target, cv, title="SVM Linear", ylim=ylim)

    # Test Naive Bayes
    estimator = MultinomialNB()
    pipeline_default = Pipeline(pp_default + [('cls', estimator)])
    pipeline_woidf = Pipeline(pp_woidf + [('cls', estimator)])
    plot_preprocessing(figure, [1, 3, 3], [pipeline_default, pipeline_woidf], ["idf = true", "idf = false"], ["g", "r"],
                       data, target, cv, title="Naive Bayes", ylim=ylim)


    figure.tight_layout()
    print('ended')


async def evaluate_logreg_classifier():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=20, test_size=0.8, random_state=17)
    ylim = [0.3, 1.0]

    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(7, 5)

    # Test Logistic Regression
    # Test Preprocessing
    estimator = LogisticRegression(random_state=42, C=2)
    train_and_plot(figure, [1, 1, 1], Pipeline(prepro + [('cls', estimator)]), data, target, "Default - Logistic Regression - C=2", cv, ylim=ylim)


    figure.tight_layout()
    print('ended')


async def evaluate_logreg_classifier_param_c():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.3, 1.0]

    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 10)

    # Test Logistic Regression
    # Test Preprocessing
    estimator025 = LogisticRegression(random_state=42, C=0.25)
    estimator05 = LogisticRegression(random_state=42, C=0.5)
    estimator1 = LogisticRegression(random_state=42)
    estimator2 = LogisticRegression(random_state=42, C=2)
    estimator4 = LogisticRegression(random_state=42, C=4)
    train_and_plot(figure, [2, 3, 1], Pipeline(prepro + [('cls', estimator025)]), data, target, "Logistic Regression - C=0.25", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 2], Pipeline(prepro + [('cls', estimator05)]), data, target, "Logistic Regression - C=0.5", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 3], Pipeline(prepro + [('cls', estimator1)]), data, target, "Logistic Regression - C=1", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 4], Pipeline(prepro + [('cls', estimator2)]), data, target, "Logistic Regression - C=2", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 5], Pipeline(prepro + [('cls', estimator4)]), data, target, "Logistic Regression - C=4", cv, ylim=ylim)


    figure.tight_layout()
    print('ended')

async def evaluate_logreg_classifier_param_tol():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.3, 1.0]
    ylim = None
    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 10)

    # Test Logistic Regression
    # Test Preprocessing
    estimator1 = LogisticRegression(random_state=42, tol=1e-10)
    estimator2 = LogisticRegression(random_state=42, tol=1e-5)
    estimator3 = LogisticRegression(random_state=42)
    estimator4 = LogisticRegression(random_state=42, tol=1e-3)
    estimator5 = LogisticRegression(random_state=42, tol=1)
    train_and_plot(figure, [2, 3, 1], Pipeline(prepro + [('cls', estimator1)]), data, target, "Logistic Regression - tol=1e-6", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 2], Pipeline(prepro + [('cls', estimator2)]), data, target, "Logistic Regression - tol=1e-5", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 3], Pipeline(prepro + [('cls', estimator3)]), data, target, "Logistic Regression - tol=1e-4", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 4], Pipeline(prepro + [('cls', estimator4)]), data, target, "Logistic Regression - tol=1e-3", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 5], Pipeline(prepro + [('cls', estimator5)]), data, target, "Logistic Regression - tol=1e-2", cv, ylim=ylim)


    figure.tight_layout()
    print('ended')


async def evaluate_logreg_classifier_param_dual():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.3, 1.0]
    ylim = None
    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 10)

    # Test Logistic Regression
    # Test Preprocessing
    estimator1 = LogisticRegression(random_state=42)
    estimator2 = LogisticRegression(random_state=42, dual=True)
    train_and_plot(figure, [2, 3, 1], Pipeline(prepro + [('cls', estimator1)]), data, target, "Logistic Regression - dual=False", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 2], Pipeline(prepro + [('cls', estimator2)]), data, target, "Logistic Regression - dual=True", cv, ylim=ylim)


    figure.tight_layout()
    print('ended')


async def evaluate_logreg_classifier_param_solver():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.4, 1.0]
    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 10)

    # Test Logistic Regression
    # Test Preprocessing
    estimator1 = LogisticRegression(random_state=42, solver='liblinear')
    estimator2 = LogisticRegression(random_state=42, solver='sag')
    estimator3 = LogisticRegression(random_state=42, solver='newton-cg')
    estimator4 = LogisticRegression(random_state=42, solver='lbfgs')
    train_and_plot(figure, [2, 3, 1], Pipeline(prepro + [('cls', estimator1)]), data, target, "Logistic Regression - solver=liblinear", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 2], Pipeline(prepro + [('cls', estimator2)]), data, target, "Logistic Regression - solver=sag", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 3], Pipeline(prepro + [('cls', estimator3)]), data, target, "Logistic Regression - solver=newton-cg", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 4], Pipeline(prepro + [('cls', estimator4)]), data, target, "Logistic Regression - solver=lbfgs", cv, ylim=ylim)


    figure.tight_layout()
    print('ended')


async def evaluate_logreg_classifier_param_penalty():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.3, 1.0]
    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(13, 5)

    # Test Logistic Regression
    # Test Preprocessing
    estimator1 = LogisticRegression(random_state=42, penalty='l2')
    estimator2 = LogisticRegression(random_state=42, penalty='l1')
    train_and_plot(figure, [1, 2, 1], Pipeline(prepro + [('cls', estimator1)]), data, target, "Logistic Regression - penalty=l2", cv, ylim=ylim)
    train_and_plot(figure, [1, 2, 2], Pipeline(prepro + [('cls', estimator2)]), data, target, "Logistic Regression - penalty=l1", cv, ylim=ylim)


    figure.tight_layout()
    print('ended')


async def evaluate_svm_classifier_param_c():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data()
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.2, random_state=17)
    ylim = [0.3, 1.0]

    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 10)

    # Test Logistic Regression
    # Test Preprocessing
    estimator025 = SVC(kernel='linear', random_state=42, C=0.25)
    estimator05 = SVC(kernel='linear', random_state=42, C=0.5)
    estimator1 = SVC(kernel='linear', random_state=42)
    estimator2 = SVC(kernel='linear', random_state=42, C=2)
    estimator4 = SVC(kernel='linear', random_state=42, C=4)
    train_and_plot(figure, [2, 3, 1], Pipeline(prepro + [('cls', estimator025)]), data, target, "SVM linear - C=0.25", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 2], Pipeline(prepro + [('cls', estimator05)]), data, target, "SVM linear - C=0.5", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 3], Pipeline(prepro + [('cls', estimator1)]), data, target, "SVM linear - C=1", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 4], Pipeline(prepro + [('cls', estimator2)]), data, target, "SVM linear - C=2", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 5], Pipeline(prepro + [('cls', estimator4)]), data, target, "SVM linear - C=4", cv, ylim=ylim)

    figure.tight_layout()
    print('ended')

async def evaluate_svm_classifier_param_gamma():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.5, 1.0]
    ylim = None

    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 10)

    # Test Logistic Regression
    # Test Preprocessing
    estimator0 = SVC(random_state=42, C=2)
    estimator1 = SVC(random_state=42, C=2, gamma=1e-2)
    estimator2 = SVC(random_state=42, C=2, gamma=0.1)
    estimator3 = SVC(random_state=42, C=2, gamma=0.25)
    estimator4 = SVC(random_state=42, C=2, gamma=0.5)
    estimator5 = SVC(random_state=42, C=2, gamma=0.75)
    train_and_plot(figure, [3, 3, 1], Pipeline(prepro + [('cls', estimator0)]), data, target, "SVM rbf - gamma=auto, C=2",
                   cv, ylim=ylim)
    train_and_plot(figure, [3, 3, 2], Pipeline(prepro + [('cls', estimator1)]), data, target, "SVM rbf - gamma=0.01, C=2", cv, ylim=ylim)
    train_and_plot(figure, [3, 3, 3], Pipeline(prepro + [('cls', estimator2)]), data, target, "SVM rbf - gamma=0.1, C=2",
                   cv, ylim=ylim)
    train_and_plot(figure, [3, 3, 4], Pipeline(prepro + [('cls', estimator3)]), data, target, "SVM rbf - gamma=0.25, C=2",
                   cv, ylim=ylim)
    train_and_plot(figure, [3, 3, 5], Pipeline(prepro + [('cls', estimator4)]), data, target, "SVM rbf - gamma=0.5, C=2",
                   cv, ylim=ylim)
    train_and_plot(figure, [3, 3, 6], Pipeline(prepro + [('cls', estimator5)]), data, target, "SVM rbf - gamma=0.75, C=2",
                   cv, ylim=ylim)

    estimator7 = SVC(random_state=42, C=0.25, gamma=0.5)
    estimator8 = SVC(random_state=42, C=0.5, gamma=0.5)
    estimator9 = SVC(random_state=42, C=1, gamma=0.5)

    train_and_plot(figure, [3, 3, 7], Pipeline(prepro + [('cls', estimator7)]), data, target, "SVM rbf - gamma=0.5, C=0.25",
                   cv, ylim=ylim)
    train_and_plot(figure, [3, 3, 8], Pipeline(prepro + [('cls', estimator8)]), data, target, "SVM rbf - gamma=0.5, C=0.5",
                   cv, ylim=ylim)
    train_and_plot(figure, [3, 3, 9], Pipeline(prepro + [('cls', estimator9)]), data, target, "SVM rbf - gamma=0.5, C=1",
                   cv, ylim=ylim)

    figure.tight_layout()
    print('ended')


async def evaluate_svm_classifier_param_kernel():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.2, random_state=17)
    ylim = [0.3, 1.0]

    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 10)

    # Test Logistic Regression
    # Test Preprocessing
    estimator1 = SVC(kernel='linear', random_state=42)
    estimator2 = SVC(kernel='poly', random_state=42)
    estimator3 = SVC(kernel='rbf', random_state=42)
    estimator4 = SVC(kernel='sigmoid', random_state=42)
    train_and_plot(figure, [2, 3, 1], Pipeline(prepro + [('cls', estimator1)]), data, target, "SVM linear", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 2], Pipeline(prepro + [('cls', estimator2)]), data, target, "SVM poly", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 3], Pipeline(prepro + [('cls', estimator3)]), data, target, "SVM rbf", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 4], Pipeline(prepro + [('cls', estimator4)]), data, target, "SVM sigmoid", cv, ylim=ylim)

    figure.tight_layout()
    print('ended')


async def evaluate_nb_classifier_param_alpha():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.3, 1.0]
    ylim = None
    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(20, 10)

    # Test Logistic Regression
    # Test Preprocessing
    estimator1 = MultinomialNB(alpha=0.25)
    estimator2 = MultinomialNB(alpha=0.5)
    estimator3 = MultinomialNB(alpha=1)
    estimator4 = MultinomialNB(alpha=2)
    estimator5 = MultinomialNB(alpha=4)
    train_and_plot(figure, [2, 3, 1], Pipeline(prepro + [('cls', estimator1)]), data, target, "Naive Bayes - alpha=0.25", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 2], Pipeline(prepro + [('cls', estimator2)]), data, target, "Naive Bayes - alpha=0.5", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 3], Pipeline(prepro + [('cls', estimator3)]), data, target, "Naive Bayes - alpha=1", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 4], Pipeline(prepro + [('cls', estimator4)]), data, target, "Naive Bayes - alpha=2", cv, ylim=ylim)
    train_and_plot(figure, [2, 3, 5], Pipeline(prepro + [('cls', estimator5)]), data, target, "Naive Bayes - alpha=4", cv, ylim=ylim)

    figure.tight_layout()
    print('ended')


async def evaluate_nb_classifier_param_prior():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data(limit=600)
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.8, random_state=17)
    ylim = [0.4, 1.0]
    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(13, 5)

    # Test Logistic Regression
    # Test Preprocessing
    estimator1 = MultinomialNB(fit_prior=True)
    estimator2 = MultinomialNB(fit_prior=False)
    train_and_plot(figure, [1, 2, 1], Pipeline(prepro + [('cls', estimator1)]), data, target, "Naive Bayes - fit_prior=True", cv, ylim=ylim)
    train_and_plot(figure, [1, 2, 2], Pipeline(prepro + [('cls', estimator2)]), data, target, "Naive Bayes - fit_prior=False", cv, ylim=ylim)

    figure.tight_layout()
    print('ended')


async def evaluate_classifier_params():
    # setup dummy classifier to load sets
    classifier = Classifier()
    classifier.training_sets = [evaluation_recording]

    # load data
    datasource = DataSource(classifier)
    data, target = await datasource.generate_fixed_evaluation_data()
    cv = ShuffleSplit(numpy.array(data).shape[0], n_iter=50, test_size=0.2, random_state=17)
    ylim = [0.6, 1.0]

    # configure pipelines
    prepro = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]

    # start plotting
    figure = pyplot.figure()
    figure.set_size_inches(13, 10)

    # Test
    estimator1 = LogisticRegression(random_state=42, C=2)
    estimator2 = SVC(random_state=42, C=2, kernel='linear')
    estimator3 = SVC(random_state=42, C=2, gamma=0.5)
    estimator4 = MultinomialNB(alpha=0.5)
    train_and_plot(figure, [2, 2, 1], Pipeline(prepro + [('cls', estimator1)]), data, target, "LogisticRegression - C=2", cv, ylim=ylim)
    train_and_plot(figure, [2, 2, 2], Pipeline(prepro + [('cls', estimator2)]), data, target, "SVM linear - C=2", cv, ylim=ylim)
    train_and_plot(figure, [2, 2, 3], Pipeline(prepro + [('cls', estimator3)]), data, target,
                   "SVM rbf - C=2, gamma=0.5", cv, ylim=ylim)
    train_and_plot(figure, [2, 2, 4], Pipeline(prepro + [('cls', estimator4)]), data, target,
                   "MultinomialNB - alpha=0.5", cv, ylim=ylim)

    figure.tight_layout()
    print('ended')


def plot_preprocessing(figure, position, classifiers, labels, colors, data, target, cv, title="", ylim=None):
    axis = figure.add_subplot(*position)
    axis.set_xlabel('# Samples')
    axis.set_ylabel('Score')

    axis.set_title(title)
    if ylim:
        axis.set_ylim(ylim)
    axis.grid()

    for idx, classifier in enumerate(classifiers):
        train_sizes, train_scores, test_scores = learning_curve(classifier, data, target, cv=cv, n_jobs=-1,
                                                                train_sizes=numpy.linspace(0.05, 1., 10))
        test_scores_mean = numpy.mean(test_scores, axis=1)
        axis.plot(train_sizes, test_scores_mean, 'o-', color=colors[idx],
                  label=labels[idx])
    axis.legend(loc="lower right")


def train_and_plot(figure, position, classifier, data, target, title, cv, ylim=None):
    train_sizes, train_scores, test_scores = learning_curve(classifier, data, target, cv=cv, n_jobs=-1, train_sizes=numpy.linspace(0.05, 1., 10))
    plot_learning_curve(figure, position, train_sizes, train_scores, test_scores, title=title, ylim=ylim)


def plot_learning_curve(figure, position, train_sizes, train_scores, test_scores, title="", ylim=None):
    axis = figure.add_subplot(*position)

    axis.set_title(title)
    if ylim:
        axis.set_ylim(ylim)
    axis.set_xlabel('# Samples')
    axis.set_ylabel('Score')

    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    axis.grid()

    axis.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    axis.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axis.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    axis.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    #axis.legend(loc="lower right")
    axis.legend(loc="best")

async def main():
    # await evaluate_preprocessing_stopwords()
    await evaluate_preprocessing_ngram()
    # await evaluate_preprocessing_idf()

    # await evaluate_logreg_classifier()
    # await evaluate_logreg_classifier_param_c()
    # await  evaluate_logreg_classifier_param_tol()
    # await evaluate_logreg_classifier_param_dual()
    # await evaluate_logreg_classifier_param_solver()
    # await evaluate_logreg_classifier_param_penalty()

    # await evaluate_svm_classifier_param_c()
    # await evaluate_svm_classifier_param_kernel()
    # await evaluate_svm_classifier_param_gamma()

    # await evaluate_nb_classifier_param_alpha()
    # await evaluate_nb_classifier_param_prior()

    # await evaluate_classifier_params()

    # await evaluate_active_learning()
    # await plot_last_active_learning()

    pyplot.savefig('data.png')
    pyplot.show()
    pyplot.close()

    # for i in range(0, 10):
    #     # await evaluate_logreg_classifier()
    #     await evaluate_active_learning()
    #     pyplot.savefig('data-{}.png'.format(i))
    #     pyplot.show()
    #     pyplot.clf()
    # pyplot.close()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
