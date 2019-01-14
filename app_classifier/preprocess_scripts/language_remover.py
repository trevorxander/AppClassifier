import pandas as pd
from polyglot.detect import Detector


# Tests for languages in specified column of csv file and removes row
#   if language is not in accepted languages
def lang_remove_csv(accepted_langs: list, csv_file: str, test_col: str):
    dataset = pd.read_csv(csv_file)
    to_remove = set()

    for index, row in dataset.iterrows():
        try:
            languages = Detector(row[test_col]).languages
            for language in languages:
                if language.name not in accepted_langs:
                    to_remove.add(index)
        except:
            to_remove.add(index)
            continue

    print('{number} of rows removed'.format(number=len(to_remove)))
    for stuff in to_remove:
        print(dataset.loc[stuff])
    dataset.drop(to_remove, inplace=True)
    dataset.to_csv(csv_file, index=False)
