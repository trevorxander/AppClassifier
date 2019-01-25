from app_classifier import naive_bayes
from app_classifier.preprocess_scripts import filter_description


def _init():
    file_loc = 'app_classifier/trained_model/app_category_model'
    nb = naive_bayes.MultinomialNBClassifier()
    nb.load_model(file_loc)
    return nb


def predict_category(description):
    to_predict = filter_description(description)
    return predict_category._classifier.predict(to_predict)[0]
predict_category._classifier = _init()
