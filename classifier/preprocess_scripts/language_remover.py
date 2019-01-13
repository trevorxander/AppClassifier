import pandas as pd
import sys
from polyglot.detect import Detector

_accepted_langs = ['English', 'un']
_filepath = sys.argv[0]
_test_col = sys.argv[1]
if __name__ == "__main__":

    dataset = pd.read_csv(_filepath)
    to_remove = set()

    for index, row in dataset.iterrows():
        try:
            languages = Detector(row[_test_col]).languages
            for language in languages:
                if language.name not in _accepted_langs:
                    to_remove.add(index)
        except:
            to_remove.add(index)
            continue

    print('{number} of rows removed'.format(number=len(to_remove)))
    for stuff in to_remove:
        print(dataset.loc[stuff])
    dataset.drop(to_remove, inplace=True)
    dataset.to_csv(_filepath, index=False)
