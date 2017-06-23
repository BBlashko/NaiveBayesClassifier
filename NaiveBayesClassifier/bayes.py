#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

class MyBayesClassifier():
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for additive smoothing
        self._feat_prob = [] # do not change the name of these vars
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []
        self._cls = []

    def train(self, X, y):
        alpha_smooth = self._smooth
        self._cls, counts = np.unique(y, return_counts=True)
        Ncls, Nfeat = len(self._cls), X.shape[1]
        self._Ncls, self._Nfeat = Ncls, Nfeat
        self._feat_prob = np.zeros((Ncls, Nfeat))
        self._class_prob = np.zeros(Ncls)

        #probability that a document (row of X) is positive = 1
        self._class_prob[1] = float(counts[1]) / len(y)
        #probability that a document (row of X) is negative = 0
        self._class_prob[0] = float(counts[0]) / len(y)

        #split X matrix into 2 matrices depending of the y value
        pos_matrix = X[y[:] == 1, :]
        neg_matrix = X[y[:] == 0, :]

        #sum each class dependant matrix
        #each sum contains a row of size Nfeat, where each column is
        #the number of occurences of that feature
        pos_feat = np.sum(pos_matrix, axis=0)
        neg_feat = np.sum(neg_matrix, axis=0)

        #number of possible words in each class
        pos_feat_size = np.sum(pos_feat)
        neg_feat_size = np.sum(neg_feat)

        # #determine the probability of each feature dependant on the class
        pos_denom = float(pos_feat_size + (alpha_smooth * Nfeat))
        neg_denom = float(neg_feat_size + (alpha_smooth * Nfeat))

        #cls[1]
        pos_feat = pos_feat.astype(float) + alpha_smooth
        self._feat_prob[1] = np.divide(pos_feat, pos_denom)

        #cls[0]
        neg_feat = neg_feat.astype(float) + alpha_smooth
        self._feat_prob[0] = np.divide(neg_feat, neg_denom)

    def classify(self, doc, _class):
        trained_feature_prob = self._feat_prob[_class]
        doc_feature_probs = trained_feature_prob[doc[:] == 1]

        classification_probability = self._class_prob[_class] * np.prod(doc_feature_probs)
        return classification_probability

    def predict(self, X):
        pred = np.zeros(len(X))

        for docIndex, doc in enumerate(X):
            #classify doc against all classes
            classifications = np.zeros(len(self._cls))
            for i, _class in enumerate(self._cls):
                classifications[i] = self.classify(doc, int(_class))

            #determine best classification based on largest probability
            pred[docIndex] = classifications.argmax()

        return pred

    @property
    def probs(self):
        # please leave this intact, we will use it for marking
        return self._class_prob, self._feat_prob

"""
Here is the calling code

"""

# added for encoding issues whilst stemming
reload(sys)
sys.setdefaultencoding("utf-8")

with open('sentiment_data/rt-polarity_utf8.neg', 'r') as f:
    lines_neg = f.read().splitlines()

with open('sentiment_data/rt-polarity_utf8.pos', 'r') as f:
    lines_pos = f.read().splitlines()

''' Uncomment below to using stemming '''
# stemmer = PorterStemmer()
# for index, line in enumerate(lines_neg):
#     temp = lines_neg[index].split();
#     line = " ".join([stemmer.stem(i) for i in temp])
#     lines_neg[index] = line
#
# for index, line in enumerate(lines_pos):
#     temp = lines_pos[index].split();
#     line = " ".join([stemmer.stem(i) for i in temp])
#     lines_pos[index] = line

#loads the 1000 lines training data, 662 test data
data_train = lines_neg[0:5000] + lines_pos[0:5000]
data_test = lines_neg[5000:] + lines_pos[5000:]

y_train = np.append(np.ones((1,5000)), (np.zeros((1,5000))))
y_test = np.append(np.ones((1,331)), np.zeros((1,331)))

# You will be changing the parameters to the CountVectorizer below
vectorizer = CountVectorizer(lowercase=True, stop_words=None,  max_df=1.0, min_df=1, max_features=None,  binary=True)
X_train = vectorizer.fit_transform(data_train).toarray()
X_test = vectorizer.transform(data_test).toarray()
feature_names = vectorizer.get_feature_names()

# Used to run classifier with multiple alpha values [0.1, 3.0]
# i = 0.1
# while i <= 3.1:
clf = MyBayesClassifier(1)
clf.train(X_train,y_train);
y_pred = clf.predict(X_test)
print np.mean((y_test-y_pred)==0)
    # i += 0.1
