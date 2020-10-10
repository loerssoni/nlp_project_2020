import pandas as pd
import numpy as np
from empath import Empath
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag

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
    data = texts.apply(lambda x: [k for k, v in lexicon.analyze(x, normalize=False).items() if v > 0])
    data = data.apply(pd.Series).stack().reset_index(level=1, drop=True)
    return data


def get_category_similarities(labels, categories, how='min_depth'):
    labels = labels.unique()
    cats = pd.Series(categories.unique(), name='category')
    
    cat_synsets = cats.apply(lambda x: (wn.synsets(x)+[]))
    cat_synsets = cat_synsets[cat_synsets.apply(lambda x: len(x) != 0)]
    label_synsets = pd.Series(labels).apply(lambda x: (wn.synsets(x)))

    def path_sims(a, b):
        c = [[i.path_similarity(j) for i in a if i.path_similarity(j) is not None] for j in b]
        return max([it for sub in c for it in sub]+[0])
    def min_depth(a, b):
        c = [[i.lowest_common_hypernyms(j)[0].min_depth() for i in a \
              if len(i.lowest_common_hypernyms(j)) != 0] for j in b]
        return max([it for sub in c for it in sub]+[0])
    if how == 'min_depth':
        func = min_depth
    else:
        func = path_sims
    hypernyms = cat_synsets.apply(lambda x: label_synsets.apply(\
                                   lambda y: func(x, y)))
    hypernyms = hypernyms[hypernyms.apply(lambda x: x == hypernyms.max(1)).sum(1) == 1]
    hypernyms.columns = labels
    cats = pd.DataFrame(cats[hypernyms.index])
    cats['label'] = hypernyms.idxmax(1)
    return cats

def wup_similarities(data):
    text_tags = data.apply(lambda x: [i[0] for i in pos_tag(word_tokenize(x.text)) if i[1] == 'NN'], 1)
    label_syns = [wn.synsets(label, pos='n')[0] for label in data.label.unique()]
    def lambda_wup_sims(label_syns, x):
        if len(x) == 0:
            sim = 0
        else:
            sim = [np.mean([label_syn.wup_similarity(wn.synsets(word, pos='n')[0])\
                if len(wn.synsets(word, pos='n')) > 0 else 0 for word in x])\
                for label_syn in label_syns]
        return pd.Series(sim, index=data.label.unique())
    wup_sims = text_tags.apply(lambda x: lambda_wup_sims(label_syns, x))
    return wup_sims