import re
regex = re.compile('[^a-zA-Z ]')

def remove_non_alpha(text):
    processed = re.sub(regex, ' ', text)
    return processed
