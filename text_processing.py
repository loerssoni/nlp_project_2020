import pandas as pd
import numpy as np
from empath import Empath

def get_category(label, categories, harv_inquirer, not_categories = []):
    cond = True
    for cat in categories:
        cond = cond&~harv_inquirer[cat].isna()
    if len(not_categories) > 0:
        for not_cat in not_categories:
            cond = cond&harv_inquirer[not_cat].isna()
    words_t = harv_inquirer.loc[cond,'Entry'].str.lower()
    words_t = pd.DataFrame(words_t.str.split('#').str[0])
    words_t['label'] = label
    words_t.columns = words_t.columns.str.lower()
    words_t.drop_duplicates(inplace=True)
    return words_t

def get_all_cat(category_dict, harv_inquirer):
    df = pd.DataFrame(columns=['entry','label'])
    for label, categories in category_dict.items():
        df = df.append(get_category(label, categories['include'],
			harv_inquirer, categories['exclude']))
    return df.reset_index(drop = True)

def process_lexicon(texts):
    lexicon = Empath()
    data = {}

    for i, text in texts.iteritems():
        data[i] = [k for k, v in lexicon.analyze(text, normalize=False).items() if v > 0]

        print("{:<5}%".format(round(i * 100 / len(texts), 2)), end='\r')
    return data