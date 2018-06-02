# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:18:43 2017

@author: ashirley
"""
import sys
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import matplotlib.pyplot as plt

class YY:
    def __init__(self, hat, test):
        self.test = test
        self.hat = hat

def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./HTRU_2.csv', header = None)
    x = data.iloc[:, :8]   
    y = data.iloc[:, 8]
    return sklearn.model_selection.train_test_split(x, y, test_size = 1 - train_ratio, random_state=0)

def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

def gradient_ascend(X, y, alpha = .001, iters = 100000, eps=.001):
    n, d = X.shape
    theta = numpy.matrix(numpy.zeros((d , 1)))
    X = numpy.matrix(X)
    y = numpy.transpose(numpy.matrix(y))
    yh = numpy.matrix(numpy.zeros((n , 1)))
    l = 1 #lambda   
    for iter in range(iters):
        e = (-1) * X * theta       
        yh = 1.0 / (1 + numpy.exp(e))        
        g = numpy.transpose(X) * (y - yh) - l/2
        diff = alpha * g
        theta = theta + diff 
        k = max(abs(diff[j]) for j in range(d))
        if k < eps:
            return theta
    return theta

def roc(y):
    fpr= numpy.array([])
    tpr= numpy.array([])
    summ = sum(y[i].test for i in range(len(y)))
    n= float(len(y) - summ)
    p= float(summ)
    TP=0.0
    FP=0.0
    for i in range(len(y)):
        tpr = numpy.insert(tpr, 0, TP/p)
        fpr = numpy.insert(fpr, 0, FP/n)
        TP = TP + y[i].test
        FP = i - TP      
    return fpr, tpr

def predict(X, theta):
    e = (-1) * X * theta       
    y = 1 / (1 + numpy.exp(e))
    return y

def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)
    X_train_scale = numpy.concatenate((numpy.ones((len(X_train_scale),1)), X_train_scale), axis = 1)
    X_test_scale = numpy.concatenate((numpy.ones((len(X_test_scale),1)), X_test_scale), axis = 1)
    theta = gradient_ascend(X_train_scale, y_train)
    y_hat = predict(X_test_scale, theta)
    y_test = list(y_test)
    y_hat.tolist()
    
    y =[]
    for i in range(len(y_test)):
        y.append(YY(y_hat[i], y_test[i]))
    y.sort(key = lambda y: y.hat, reverse = True)        
    fpr, tpr = roc(y)
    plt.plot(fpr, tpr)
    plt.show()

if __name__ == "__main__":
    main(sys.argv)