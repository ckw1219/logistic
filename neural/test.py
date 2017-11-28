# -*- coding: utf-8 -*-
__author__ = 'Administrator'

import numpy as np

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -(1 / m) * np.sum((Y *np.log(A) + Y *np.log(1 - A)), axis=1, keepdims=True)

    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1 / m * np.sum(A-Y, axis=1, keepdims=True)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
            "db": db}

    return grads, cost
'''
print("dw = "+str(grads["dw"]))
print("db = "+str(grads["db"]))
print("cost ="+str(cost))
'''

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert (w.shape == (dim,1))
    assert (isinstance(b,float) or isinstance(b,int))
    return w,b
#axis=1以后就是将一个矩阵的每一行向量相加

'''
print("w ="+str(w))
print("b = "+str(b))
'''
def optimize(w,b,X,Y,num_iterations,learn_rate,print_cost=False):

    costs = []
    for i in range(num_iterations):

        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w- learn_rate*dw
        b = b- learn_rate*db

        if i%100 ==0:
            costs.append(cost)
        if print_cost and i%100 ==0:
            print("Cost after iteratuon %i : %f" %(i,cost))

        params = {"w" : w,
                  "b" : b}
        grads = {"dw" : dw,
                 "db" : db}
        return params,grads,costs

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        if A[0][i] >= 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
    assert (Y_prediction.shape == (1,m))

    return Y_prediction

X = np.array([[1,2],[3,4]])
w = np.array([0.15,0.45])
b = 0.021
T = predict(w,b,X)
print(T)

def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000, learn_rate = 0.5,print_cost = False):
    w,b = initialize_with_zeros(dim)
    parameters,grads,costs = propagate(w,b,X_train,Y_train)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    print("train accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_test-Y_test))*100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learn_rate,
         "num_iteration": num_iterations}
    return d



