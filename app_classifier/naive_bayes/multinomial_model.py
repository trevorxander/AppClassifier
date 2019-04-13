import math
import pickle


class Model:
    def __init__(self):
        self._SMOOTHING = 1 * 10 ** -15
        self._category_prob = {}
        self._feature_set = set()
        self._category_features_freq = {}

    def train(self, x, y):
        self._category_probability(y)
        self._feature_count(x, y)

    def store_model(self, file_loc):
        model_file = open(file_loc, 'wb')
        pickle.dump(self._category_prob, model_file)
        pickle.dump(self._feature_set, model_file)
        pickle.dump(self._category_features_freq, model_file)
        model_file.close()

        # print('Vocabulary = {vocab}\n'.format(vocab=self._feature_set))
        # for category, [prob, count] in self._category_prob.items():
        #     print('C({category}) = {count}\nP({category}) = {prob}'.format(category=category, count=count, prob=prob))
        # print('\n')
        #
        # for (category, feature), count in self._category_features_freq.items():
        #     print('C({category},{feature}) = {count}'.format(category=category, feature=feature, count=count))
        # print('\n')

    def load_model(self, file_loc=None):
        model_file = open(file_loc, 'rb')
        self._category_prob = pickle.load(model_file)
        self._feature_set = pickle.load(model_file)
        self._category_features_freq = pickle.load(model_file)
        model_file.close()

    def evaluate(self, x, y):
        correct_prediction = 0
        wrong_prediction = 0
        for test_feature, expected_category in zip(x, y):
            predicted_category = self.predict(test_feature)
            # print('expected_category({features}) = {category}\n'.format(category=expected_category,
            #                                                            features=test_feature))
            if predicted_category == expected_category:
                correct_prediction += 1
            else:
                wrong_prediction += 1

        error = wrong_prediction / (correct_prediction + wrong_prediction)

        return error

    def predict(self, features):
        most_probable_category = None
        highest_probability = float('-inf')
        for category in self._category_prob.keys():
            probability = self.log_likelihood(category, features) + self.log_prior(category)
            if probability > highest_probability:
                highest_probability = probability
                most_probable_category = category

        # print('predicted_category({features}) = {category}'.format(category=most_probable_category,
        #                                                            features=features))
        return most_probable_category

    def log_likelihood(self, category, features):
        total_log_conditional = 0
        for feature in features.split():
            total_log_conditional += self.log_conditional(feature, category)
        return self.log_prior(category) + total_log_conditional

    def log_prior(self, category):
        return math.log(self._category_prob[category][0])

    def log_conditional(self, feature, category):
        if (category, feature) not in self._category_features_freq:
            feature_freq = 0
        else:
            feature_freq = self._category_features_freq[category, feature]

        category_count = self._category_prob[category][1]

        prob = feature_freq + self._SMOOTHING / \
               category_count + (len(self._feature_set) * self._SMOOTHING)
        return math.log(prob)

    def _category_probability(self, category_set):
        total = 0
        for category in category_set:
            total += 1
            if category in self._category_prob.keys():
                self._category_prob[category][0] += 1
                self._category_prob[category][1] += 1
            else:
                self._category_prob[category] = [1, 1]

        for category in self._category_prob.keys():
            self._category_prob[category][0] /= total

        # print(self._category_prob)


    def _feature_count(self, feature_set, category_set):
        for features, category in zip(feature_set, category_set):
            for feature in features.split():
                self._feature_set.add(feature)
                if (category, feature) not in self._category_features_freq:
                    self._category_features_freq[category, feature] = 1
                else:
                    self._category_features_freq[category, feature] += 1
