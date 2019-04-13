import pandas as pd
from app_classifier import preprocess_scripts as ps
from sklearn.model_selection import train_test_split
from app_classifier.naive_bayes import Model


def train_from_csv(csv_file, save_model_file, label_col, feature_col, ngram=2):
    dataset = pd.read_csv(csv_file)
    y = dataset[label_col]
    x = dataset[feature_col].apply(ps.preprocess)
    x = x.apply(lambda x: ps.ngrams(x, ngram))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    nb = Model()
    nb.train(x_train, y_train)

    nb.store_model(save_model_file)
    print('Error: ', nb.evaluate(x_test, y_test) * 100)

    return nb

def load_trained_model (model_file):
    nb = Model()
    nb.load_model(model_file)
    return nb