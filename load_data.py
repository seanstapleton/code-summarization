import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext import data
import spacy
import pandas as pd
import re
import tqdm
from torchtext import vocab
from torch.autograd import Variable


spacy_en = spacy.load('en')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(clean(text))]

def clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    return text.strip()

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
# LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

train_val_fields = [
    ('id', None), # process it as label
    ('Text', TEXT) # process it as text
]

# pred_val_fields = [
#     ('id', None),
#     ('SentimentText', TEXT) # process it as text
# ]

# trainds, valds, testds = data.TabularDataset.splits(path='./data', 
#                                                     format='tsv', 
#                                                     train='data_ps.descriptions.train.tsv', 
#                                                     validation='data_ps.descriptions.valid.tsv',
#                                                     test='data_ps.descriptions.test.tsv',
#                                                     fields=train_val_fields, 
#                                                     skip_header=True)

testds = data.TabularDataset.splits(path='./data', 
                                    format='tsv', 
                                    test='data_ps.descriptions.test.tsv', 
                                    fields=train_val_fields, 
                                    skip_header=True)

# unlabds = data.TabularDataset(path='./data/unlabelled.tsv', 
#                                     format='tsv', 
#                                     fields=pred_val_fields, 
#                                     skip_header=True)

# vec = vocab.Vectors('glove.840B.300d.txt', './data')

# TEXT.build_vocab(trainds, max_size=100000, vectors=vec)
TEXT.build_vocab(trainds, max_size=100000)
LABEL.build_vocab(trainds)