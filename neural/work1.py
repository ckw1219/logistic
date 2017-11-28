# -*- coding: utf-8 -*-
__author__ = 'Administrator'

import math
import numpy as np


def basic_sigmoid(x):
    s=1/(1+math.exp(-x))
    return s

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

def image2vector(image):
    v=v.reshape((v.shape[0]*v.shape[1],v.shape[2]))
    return v

def normalizeRows(x):
    x_norm = np.linalg.norm(x,axis=1,keepdims=True)
    x = x/x_norm
    return x

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    s = x_exp/x_sum
    return s

def L1(yhat,y):
    loss = np.sum(np.abs(yhat-y),axis=0,keepdims=True)
    return loss
x = np.array([[1,2,3],[4,5,6]])

yhat = np.array([0.9,0.2,0.1,0.4,0.9])
y = np.array([1,0,0,1,1])

x1 = np.array([1,2,3,4])
x2 = np.array([3,4,5])

w = np.array([[1],[2]])
b = 2
X = np.array([[1,2],[3,4]])
A = np.dot(w.T,X)+b
print(A)
