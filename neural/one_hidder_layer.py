# -*- coding: utf-8 -*-
__author__ = 'Administrator'

'''
1.Implement a 2-class classification neural network with a single hidder layer 
2.Use units with a non-linear activation function,such as tanh
3.Compute the cross entropy loss
4.Implement forward and backward propagation
'''

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases import *
from planar_utils import plot_decision_boundary,sigmoid, load_planar_dataset, load_extra_datasets

X,Y = load_planar_dataset()
# a numpy-array X that contains features(x1,x2)
# a numpy-array Y that contains labels (red:0,blue:1)
#plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)
#plt.show()
shape_X = np.shape(X)
shape_Y = np.shape(Y)
#m = shape_X[1]
'''
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)
plot_decision_boundary(lambda x: clf.predict(x),X,Y)
plt.title("LG")
plt.show()
'''


'''
1.Define the neural network structure
2.Initialize the model's parameters
3.Loop:
    Implement forward propagation
    Compute loss
    Implement backwark propagation to get the gradients
    Update parameters(gradient descent)
'''
def layer_size(X,Y):
    '''
    X--input dataset of shape(input size,number of examples)
    Y--labels of shape(output size,number of examples)

    Return:
        n_x -- the size of the input layer
        n_h -- the size of the hidder layer
        n_y -- the size of output layer
    '''
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x,n_h,n_y)
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    assert (W1.shape == (n_h,n_x))
    assert (b1.shape == (n_h,1))
    assert (W2.shape == (n_y,n_h))
    assert (b2.shape == (n_y,1))

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

def forward_propagation(X,parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]#size(n_h,n_x)
    b1 = parameters["b1"]
    W2 = parameters["W2"]#size(n_y,n_h)
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X)+b1#n_h*m
    A1 = np.tanh(Z1)#n_h*m
    Z2 = np.dot(W2,A1)+b2#n_y*m
    A2 = sigmoid(Z2)#

    assert (A2.shape == (1,X.shape[1]))
    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }
    return A2,cache



def compute_cost(A2,Y,parameters):
    """

    :param A2: the sigmoid output of the second activation of shape(1,num of examples)
    :param Y:"true" labels vector of shape(1,num of examples)
    :param parameters:
    :return:cross-entropy cost given equation
    """

    m = Y.shape[1]# number of examples
    logprobs = np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)
    cost = -np.sum(logprobs)*(1/m)

    cost = np.squeeze(cost)

    assert (isinstance(cost,float))

    return cost
def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dz2 = A2-Y
    dw2 = 1/m*np.dot(dz2,A1.T)
    db2 = 1 / m * np.sum(A2 - Y, axis=1, keepdims=True)
    dz1 = np.dot(W2.T,dz2)*(1-np.power(A1,2))
    dw1 = 1/m*np.dot(dz1,X.T)
    db1 = 1/m*np.sum(dz1,axis=1,keepdims=True)

    grads = {
        "dw2":dw2,
        "db2":db2,
        "dw1":dw1,
        "db1":db1
    }
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dw1 = grads["dw1"]
    db1 = grads["db1"]
    dw2 = grads["dw2"]
    db2 = grads["db2"]

    W1 = W1-learning_rate*dw1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dw2
    b2 = b2-learning_rate*db2

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
    }
    return parameters


def nn_model(X,Y,n_h,num_iterations = 10000,print_cost = False):
    '''

    :param X: dataset of shape(2,num of examples)
    :param Y: labels of shape(1,num of examples)
    :param n_h: size of hidden layer
    :param num_iterations: Num of iteration in gradient descent loop
    :param print_cost:if true,print the cost every 1000 iterations
    :return:parameters learnt by the model.
    '''

    np.random.seed(3)
    n_x = layer_size(X,Y)[0]
    n_y = layer_size(X,Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0,num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads)

        if print_cost and i %1000 ==0:
            print("Cost after iteration %i: %f" %(i,cost))

    return parameters

def predict(parameters,X):
    A2,cache = forward_propagation(X,parameters)
    predictions = np.array([0 if i <= 0.5 else 1 for i in np.squeeze(A2)])

    return predictions

parameters = nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=False)
plot_decision_boundary(lambda x: predict(parameters,x.T),X,Y)
plt.title("Decision Boundary for hidden layer size"+str(4))
plt.show()
predictions = predict(parameters,X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')