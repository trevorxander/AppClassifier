from app_classifier import naive_bayes

file_name = 'dataset/apple_appstore.csv'
model_file = 'trained_models/bigram_model'

if __name__ == "__main__":
    nb = naive_bayes.train_from_csv(file_name, model_file, 'category', 'description', ngram=3)
