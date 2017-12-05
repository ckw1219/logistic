# -*- coding: utf-8 -*-
__author__ = 'Administrator'

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import  Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()
#the orig is meaning that the preproessing
#index = 23
#plt.imshow(train_set_x_orig[index])
#print("y = "+str(train_set_y[:,index])+", it's a "+classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+" picture.")
#plt.show()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


'''
print("Number of training examples: m_train ="+str(m_train))
print("Number of testing examples: m_test = "+str(m_test))
print("Height/Width of each image: num_px"+str(num_px))
print("Each image is of size: ("+str(num_px)+","+str(num_px),",3)")
print("train_set_x_orig shape:"+str(train_set_x_orig.shape))
print("train_set_y shape:"+str(train_set_y.shape))
print("test_set_x shape:"+str(test_set_x_orig.shape))
print("test_set_y shape:"+str(test_set_y.shape))
'''
#train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0])
#test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3],test_set_x_orig.shape[0])
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
'''
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
'''

train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0

'''
Common steps for pre-processing a new dataset are:
  one:Figure out the dimensions and shapes of the problem(m_train,m_tett,num_px)
  two:Reshape the datasets such that each example is now a vector of size(num_px*num_px*c,1)
  three: Standardize the data
'''


def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

#print("sigmoid([0,2]) = " + str(sigmoid(np.array([0,2]))))

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert (w.shape == (dim,1))
    assert (isinstance(b,float) or isinstance(b,int))
    return w,b
#axis=1以后就是将一个矩阵的每一行向量相加

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -(1 / m) * np.sum((Y *np.log(A) + (1-Y)*np.log(1 - A)), axis=1, keepdims=True)

    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1 / m * np.sum(A-Y, axis=1, keepdims=True)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
            "db": db}

    return grads, cost

#w,b,X,Y = np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
#grads,cost = propagate(w,b,X,Y)
#print(grads["dw"])
#print(grads["db"])
#print(cost)

def optimize(w,b,X,Y,num_iterations,learn_rate,print_cost=False):

    costs = []
    for i in range(num_iterations):

        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        w =w - learn_rate*dw
        b =b - learn_rate*db

        if i%100 ==0:
            costs.append(cost)
        if print_cost and i%100 ==0:
            print("Cost after iteratuon %i : %f" %(i,cost))

        params = {"w" : w,
                  "b" : b}
        grads = {"dw" : dw,
                 "db" : db}
    return params,grads,costs
#params,grads,costs = optimize(w,b,X,Y,num_iterations=100,learn_rate=0.009,print_cost= False)

'''
print("w ="+str(params["w"]))
print("b ="+str(params["b"]))
print("dw ="+str(grads["dw"]))
print("db ="+str(grads["db"]))
'''
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
#print("prediction =" + str(predict(w,b,X)))

def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000, learn_rate = 0.5,print_cost = False):
    w,b = initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learn_rate,print_cost=False)
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test).reshape(1,X_test.shape[1])
    Y_prediction_train = predict(w,b,X_train).reshape(1,X_train.shape[1])

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

d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learn_rate=0.005,print_cost=True)
'''
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds')
plt.title("Learning rate ="+str(d["learning_rate"]))
plt.show()
'''

'''
learning_rates = [0.01,0.001,0.0001]
models = {}
for i in learning_rates:
    print("learning rate is :"+str(i))
    models[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=1500,learn_rate=i,print_cost=True)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]["learning_rate"]))

plt.ylabel("cost")
plt.xlabel("iterations")

legend = plt.legend(loc = 'upper center',shadow =True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
'''

my_image ="my_image.jpg"
fname = "images/"+my_image
image = np.array(ndimage.imread(fname,flatten=False))
my_image = scipy.misc.imresize(image,size=(num_px,num_px)).reshape((1,num_px*num_px*3)).T
my_predicted_image = predict(d["w"],d["b"],my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

