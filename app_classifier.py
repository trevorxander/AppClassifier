from app_classifier import naive_bayes
from app_classifier.preprocess_scripts import filter_description

if __name__ == "__main__":
    file_loc = 'app_classifier/trained_model/app_category_model'
    nb = naive_bayes.MultinomialNBClassifier()
    nb.load_model(file_loc)

    to_predict = filter_description('workout exercise strength muscle')
    print(nb.predict(to_predict))
