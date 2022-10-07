# -*- coding: utf-8 -*-
"""
Created on Tue nov 12 14:14:06 2019

@author: YTX

"""
from sklearn.naive_bayes import GaussianNB


def gaussianNB_classifier_cal(dx,dy,priors,var_smoothing):
    X = dx.values
    Y = dy.values
    clf = GaussianNB(priors,var_smoothing)
    clf.fit(X,Y)
    prediction = clf.predict(X)
    score = clf.score(X,prediction)
    return score
