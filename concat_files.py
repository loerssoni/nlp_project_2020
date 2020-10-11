import pandas as pd
train = pd.read_csv('./data/train.txt', header=None, names=['text','label'], sep=';')
test = pd.read_csv('./data/test.txt', header=None, names=['text','label'], sep=';')
val = pd.read_csv('./data/val.txt', header=None, names=['text','label'], sep=';')
data = pd.concat([train, test, val])
data.reset_index(drop=True, inplace=True)
data.to_csv('./data/full_data.txt', header=None, sep=';')