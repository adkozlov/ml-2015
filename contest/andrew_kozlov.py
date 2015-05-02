#!/usr/bin/env python3

from nltk import SnowballStemmer, RegexpTokenizer
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from contest.features import SwearWordsTransformer


__author__ = 'Andrew Kozlov'
__copyright__ = 'Copyright 2015, SPbAU'


def read_data_set(file_name, columns_count):
    with open(file_name) as file:
        return [''.join(c for c in line if ord(c) < 128).split(',', columns_count - 1) for line in file.readlines()]


def split_data_set(xs, x_index, y_index):
    return [x[x_index].replace('\"', '').strip() for x in xs], [int(x[y_index]) for x in
                                                                        xs] if y_index else None


def write_predicted(file_name, ys):
    result = np.transpose(np.vstack((np.arange(len(ys)), ys)))
    np.savetxt(fname=file_name, X=result, fmt='%d', delimiter=',', header='Id,Insult', comments='')


def split_with_ratio(xs, ys, ratio=0.8):
    bound = int(ys.size * ratio)
    return xs[:bound], ys[:bound], xs[bound:], ys[bound:]


def update_words(xs):
    stemmer = SnowballStemmer('english', ignore_stopwords=False)
    tokenizer = RegexpTokenizer(r'\w+')
    return [' '.join([stemmer.stem(token) for token in tokenizer.tokenize(x)]) for x in xs]


def my_f_regression(x, y):
    return f_regression(x, y, False)


def my_auc(x, y):
    return auc(x, y, True)


if __name__ == '__main__':
    x_train, y_train = split_data_set(read_data_set('train_au.csv', 3)[1:], 2, 1)
    x_train = update_words(x_train)
    y_train = np.array(y_train)

    parameters = {
        'words_model__vectorizer__ngram_range': [(1, bound) for bound in range(5, 6)],
        'words_model__vectorizer__min_df': list(range(3, 4)),
        'percentile__score_func': [f_classif],
        'percentile__percentile': [5 * i for i in range(4, 5)],
        'sgd__loss': ['squared_hinge'],
        'sgd__alpha': [10 ** (-power) for power in range(7, 8)],
        'sgd__epsilon': [10 ** (-power) for power in range(4, 5)],
        'sgd__penalty': ['l2']
    }
    score_func = my_auc

    words_model = FeatureUnion([('vectorizer', TfidfVectorizer()),
                                ('swear_words_transformer', SwearWordsTransformer())])
    model = Pipeline([('words_model', words_model),
                      ('percentile', SelectPercentile()),
                      ('sgd', SGDClassifier())])
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, score_func=score_func)
    grid_search.fit(x_train, y_train)

    print('%s: %0.3f' % (score_func.__name__, grid_search.best_score_))
    best_parameters = grid_search.best_estimator_.get_params()
    for name in sorted(parameters.keys()):
        print('\t%s: %r' % (name, best_parameters[name]))

    grid_search.set_params()

    x_test, y_test = split_data_set(read_data_set('test_au.csv', 2), 1, None)
    x_test = update_words(x_test)
    write_predicted('submission.csv', grid_search.predict(x_test))