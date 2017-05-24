#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class MyBayesClassifier():
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for additive smoothing
        self._feat_prob = [] # do not change the name of these vars
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    def train(self, X, y):
        alpha_smooth = self._smooth
        cls = np.unique(y)
        Ncls, Nfeat = len(cls), X.shape[1]
        self._Ncls, self._Nfeat = Ncls, Nfeat
        self._feat_prob = np.zeros((Ncls, Nfeat))
        self._class_prob = np.zeros(Ncls)
        # your code goes here
        
    def predict(self, X):
        pred = np.zeros(len(X))
        # your code goes here
        return pred

    @property
    def probs(self):
        # please leave this intact, we will use it for marking
        return self._class_prob, self._feat_prob


""" 
Here is the calling code

"""

with open('sentiment_data/rt-polarity_utf8.neg', 'r') as f:
    lines_neg = f.read().splitlines()

with open('sentiment_data/rt-polarity_utf8.pos', 'r') as f:
    lines_pos = f.read().splitlines()

data_train = lines_neg[0:5000] + lines_pos[0:5000]
data_test = lines_neg[5000:] + lines_pos[5000:]

y_train = np.append(np.ones((1,5000)), (np.zeros((1,5000))))
y_test = np.append(np.ones((1,331)), np.zeros((1,331)))

# You will be changing the parameters to the CountVectorizer below
vectorizer = CountVectorizer(lowercase=True, stop_words=None,  max_df=1.0, min_df=1, max_features=None,  binary=True)
X_train = vectorizer.fit_transform(data_train).toarray()
X_test = vectorizer.transform(data_test).toarray()
feature_names = vectorizer.get_feature_names()

clf = MyBayesClassifier(1);
clf.train(X_train,y_train);
y_pred = clf.predict(X_test)
print np.mean((y_test-y_pred)==0)

