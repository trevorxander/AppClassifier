def ngrams(text, n):
    tokens = [token for token in text.split(" ") if token != ""]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngram_setntence = ' '.join([('_'.join(ngram)) + ' ' for ngram in ngrams])
    return ngram_setntence


if __name__ == '__main__':
    print(ngrams('This is a sentence', 2))
