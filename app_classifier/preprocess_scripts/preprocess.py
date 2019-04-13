from app_classifier.preprocess_scripts.filter import remove_non_alpha
from app_classifier.preprocess_scripts.stop_words import filter_stop_words
from app_classifier.preprocess_scripts.language_remover import lang_remove_csv

_preprocess = [str.lower, remove_non_alpha, filter_stop_words]

def preprocess(sentence):
    processed_sentence = sentence
    for process in _preprocess:
        processed_sentence = process(processed_sentence)

    return processed_sentence
