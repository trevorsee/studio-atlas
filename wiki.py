import string
import os
import time
import _pickle as pickle
import json
import wikipedia
import requests
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD


main = wikipedia.page('List of graphic designers')
token_dict = {}
for i, article in enumerate(main.links):
    if article not in token_dict:
        time.sleep(5)  # helps to avoid hangups. Ctrl-C in case you get stuck on one
        if i%60==0:
            print(article)
            print("getting text for article %d/%d : %s"%(i, len(main.links), article))
        try:
            text = wikipedia.page(article)
            token_dict[article] = text.content
        except:
            print(" ==> error processing "+article)



def tokenize(text):
    text = text.lower() # lower case
    for e in set(string.punctuation+'\n'+'\t'): # remove punctuation and line breaks/tabs
        text = text.replace(e, ' ')
    for i in range(0,10):	# remove double spaces
        text = text.replace('  ', ' ')
    text = text.translate(string.punctuation)  # punctuation
    tokens = nltk.word_tokenize(text)
    text = [w for w in tokens if not w in stopwords.words('english')] # stopwords
    stems = []
    for item in tokens: # stem
        stems.append(PorterStemmer().stem(item))
    return stems

# calculate tfidf (might take a while)
print("calculating tf-idf")
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
print("reducing tf-idf to 500 dim")
tfs_reduced = TruncatedSVD(n_components=500, random_state=0).fit_transform(tfs)
print("done")


print(tfs)
print("term 20000 = \"%s\""%tfidf.get_feature_names()[100])
