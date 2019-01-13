from sklearn.model_selection import train_test_split
import pandas as pd


class Learner:
    _categories = {}
    _features = {}
    _category_features_freq = {}

    def __init__(self, training_set):

        X_train, X_test, y_train, y_test = train_test_split(training_set,
                                                            test_size=.20,
                                                            shuffle=True)
        self.attributes = X_train
        self.labels = y_train
        self.train()

    def train(self):
        pass

    def posterior_probability(self, category, features):
        feature_likelihood = 1

        for feature in features:
            feature_likelihood *= self.liklihood(feature, category)

        self.prior_probability(category) * feature_likelihood
        pass

    def prior_probability(self, category):
        return self._categories[category]

    def liklihood(self, feature, category):
        return self._category_features_freq[category, feature] \
               / (self._categories[feature][1] + len(self._features))

    def category_probability(self):
        for category in self.labels:
            if category in self._categories.keys():
                self._categories[category] += 1
            else:
                self._categories[category] = 1

        for category in self._categories.keys():
            self._categories[category] / len(self._categories)
