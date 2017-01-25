import string
import os
import time
import _pickle as pickle
import json
import requests
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD


r = requests.get('https://sheetlabs.com/DV/studiolist', auth=('tcarr@mica.edu', 't_9e773033c7352d4e31ca378e0dc8cae7'))
main = r.json()

token_dict = {}
for i, studio in enumerate(main):
     if studio['name'] not in token_dict:
#         time.sleep(5)  # helps to avoid hangups. Ctrl-C in case you get stuck on one
         print("getting text for studio %d/%d : %s"%(i, len(main), studio['name']))
         try:
             text = studio['description']
             token_dict[studio['name']] = text
         except:
             print(" ==> error processing "+studio['name'])



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
print("term 2000 = \"%s\""%tfidf.get_feature_names()[200])

model = TSNE(n_components=2, perplexity=3, verbose=2, learning_rate=100).fit_transform(tfs.todense())
#model = TSNE(n_components=2, perplexity=10, verbose=2).fit_transform(tfs_reduced)

# save to json file
x_axis=model[:,0]
y_axis=model[:,1]
x_norm = (x_axis-np.min(x_axis)) / (np.max(x_axis) - np.min(x_axis))
y_norm = (y_axis-np.min(y_axis)) / (np.max(y_axis) - np.min(y_axis))
data = {"x":x_norm.tolist(), "y":y_norm.tolist(), "names":list(token_dict.keys())}
with open('data_studios.json', 'w') as outfile:
    json.dump(data, outfile)
