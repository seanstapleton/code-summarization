import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import string
import ast
from tqdm import tqdm

# import nltk
from scipy.stats import uniform, expon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel

import string

# from nltk import word_tokenize
# from nltk.stem.porter import PorterStemmer

# term_dict = {
#   "punct" : [",", "|", " "]
# }

def tokenize(text):
    # punct = term_dict['punct']
    # text = "".join([ch for ch in text if ch not in punct])
    # tokens = nltk.word_tokenize(text)
    # print(text)
    text = text.replace('|', ',')
    text = text.replace(' ', ',')
    # stems = stem_tokens(tokens, stemmer)
    # tokens = re.split("[,.| ]", text)
    tokens = text.split(',')
    return tokens


# count_vectorizer = CountVectorizer(tokenizer=tokenize, lowercase = False)
# tfidf = TfidfTransformer()
tfidf_vectorizer = TfidfVectorizer()

with open('{}/{}.train.c2s'.format("java-small", "java-small"), 'r') as file:
    # print(file)
    complete_data = []
    print("fitting file")
    X = tfidf_vectorizer.fit_transform(file)
feature_names = tfidf_vectorizer.get_feature_names()
# print(X)
# print(feature_names)
arr = []
for i in tqdm(range(691974)):
    feature_index = X[i,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [X[i, x] for x in feature_index])
    dct_scores = {}
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        dct_scores[w] = s
    arr.append([i, dct_scores])
arr = np.array(arr)
# print(np.array(arr))
print("saving...")

with open('java-small.train.c2s.tfidf', 'wb') as f: pickle.dump(arr, f)

with open('java-small.train.c2s.tfidf', 'rb') as f: arrayname1 = pickle.load(f)

print("sanity check", np.array_equal(arr, arrayname1))



# with open('{}.val.c2s'.format("java-small"), 'r') as file:    
#     for line in tqdm(file):
#         # print(line)
#         line = [line]
#         per_line_vec = TfidfVectorizer()
#         # count = count_vectorizer.fit_transform(line)
#         # print("count", count)
#         # val = tfidf.fit_transform(count)
#         # print("tfidf", val)
#         # complete_data.append((count, val))
#         per_line_vec.fit(line)
#         # tfidf_vectorizer.fit(line)
#         # tokens = tokenize(text)
#         # lis = []
#         # for token in tokens:
#         #     val = tfidf_vectorizer.vocabulary_
#         #     lis.append()
#         complete_data.append((per_line_vec.vocabulary_, per_line_vec.idf_))
#         print(complete_data)
#         exit()
#     print(complete_data)
    
    # print(complete_data)

    # subtoken_to_count = pickle.load(file)
    # node_to_count = pickle.load(file)
    # target_to_count = pickle.load(file)
    # max_contexts = pickle.load(file)
    # vectorizer = CountVectorizer()
    # vectorizer.fit(text)
# print(subtoken_to_count)
