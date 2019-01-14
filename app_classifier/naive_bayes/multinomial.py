from sklearn.model_selection import train_test_split
import pickle
import math


class MultinomialNBClassifier:
    _SMOOTHENING = 0.00001

    def __init__(self):
        self._category_prob = {}
        self._feature_set = set()
        self._category_features_freq = {}

    def train(self, x, y):
        self.attributes, self._attributes_test, \
        self.labels, self._labels_test = train_test_split(x, y,
                                                          test_size=.10,
                                                          shuffle=True,
                                                          random_state=0)

        self._category_probability()
        self._feature_count()
        self.cross_validation()

    def load_model(self, file_loc=None):
        model_file = open(file_loc, 'rb')
        self._category_prob = pickle.load(model_file)
        self._feature_set = pickle.load(model_file)
        self._category_features_freq = pickle.load(model_file)
        self._error = pickle.load(model_file)
        model_file.close()

    @property
    def error(self):
        return self._error

    def cross_validation(self):
        correct_prediction = 0
        wrong_prediction = 0

        for test_feature, expected_category in zip(self._attributes_test, self._labels_test):
            predicted_category = self.predict(test_feature)[0]
            if predicted_category == expected_category:
                correct_prediction += 1
            else:
                wrong_prediction += 1

        self._error = wrong_prediction / (correct_prediction + wrong_prediction)

    def store_model(self, file_loc):
        model_file = open(file_loc, 'wb')
        pickle.dump(self._category_prob, model_file)
        pickle.dump(self._feature_set, model_file)
        pickle.dump(self._category_features_freq, model_file)
        pickle.dump(self._error, model_file)
        model_file.close()

    def predict(self, features):
        probable_category = ''
        max_probablility = float('-inf')
        for category in self._category_prob.keys():
            probability = self.log_posterior_probability(category, features)
            if probability > max_probablility:
                max_probablility = probability
                probable_category = category

        return probable_category, math.exp(max_probablility)

    def log_posterior_probability(self, category, features):

        feature_likelihood = math.log(1)
        for feature in features.split():
            feature_likelihood += math.log(self.likelihood(feature.lower(), category))
        return math.log(self.prior_probability(category)) + feature_likelihood

    def prior_probability(self, category):
        return self._category_prob[category][0]

    def likelihood(self, feature, category):
        if (category, feature) not in self._category_features_freq:
            feature_freq = 0
        else:
            feature_freq = self._category_features_freq[category, feature]
        category_count = self._category_prob[category][1]

        return (feature_freq + self._SMOOTHENING) / \
               (category_count + len(self._feature_set) + self._SMOOTHENING)

    def _category_probability(self):
        total = 0
        for category in self.labels:
            total += 1
            if category in self._category_prob.keys():
                self._category_prob[category][0] += 1
            else:
                self._category_prob[category] = [1, 0]

        for category in self._category_prob.keys():
            self._category_prob[category][0] /= total

    def _feature_count(self):
        for category, features in zip(self.labels, self.attributes):
            for feature in features.split():
                self._feature_set.add(feature)
                self._category_prob[category][1] += 1
                if (category, feature) not in self._category_features_freq:
                    self._category_features_freq[category, feature] = 1
                else:
                    self._category_features_freq[category, feature] += 1
