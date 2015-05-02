import numpy as np

__author__ = 'Andrew Kozlov'
__copyright__ = 'Copyright 2015, SPbAU'


class AbstractTransformer:
    inits_counter_ = 0

    @staticmethod
    def print_counter():
        print(SwearWordsTransformer.inits_counter_)
        SwearWordsTransformer.inits_counter_ += 1

    def __init__(self):
        AbstractTransformer.print_counter()

    def fit(self, xs, ys=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, *args, **kwargs):
        return self

    @staticmethod
    def get_words_count(xs):
        return np.array([len(x) for x in xs])


class SwearWordsTransformer(AbstractTransformer):

    def __init__(self):
        super().__init__()
        with open('swear_words.txt') as file:
            self.swear_words_ = [line.strip() for line in file.readlines()]

    def transform(self, xs):
        swear_words_count = np.array(
            [np.sum([x.lower().count(swear_word) for swear_word in self.swear_words_]) for x in xs])
        swear_words_ratio = np.true_divide(swear_words_count, SwearWordsTransformer.get_words_count(xs))

        return np.array([swear_words_count, swear_words_ratio]).T

    def get_feature_names(self):
        return np.array(['swear_words_count', 'swear_words_ratio'])