#!/usr/bin/python 
import numpy as np
import copy
import sys

lr = float(sys.argv[1]) #learning rate, try 10e-4, 1.0, 10e4

def farray(matrix):
    return np.array(matrix,dtype=np.float32)

#input: [0,1] (in this case: {0,1})
X = farray([[0,0],
            [0,1],
            [1,0],
            [1,1]])
#target class: {-1,1}
T = farray([[-1],
            [1],
            [1],
            [1]])

#---initialize weights with random value---
#<Uniform distribution>
W=np.random.rand(3,1)
#<Gaussian distribution>
#W=np.random.randn(3,2)
#------------------------

def activation_func(x): #step function
    return np.sign(x)+10e-7 #if x==0; return 1.0

def perceptron_criteria(w):
    D = 0
    Y = T*forward(w,X)
    for y in Y:
        if y[0] > 0:
            D += y[0] # add loss only if miss classification happens
    return D

def add_bias(matrix):
    """
    add bias to X
    
    e.g.    [[0,0],     [[1,0,0],
             [0,1], -->  [1,0,1],
             [1,0],      [1,1,0],
             [1,1]]      [1,1,1]]
            
    """
    L = len(matrix)
    Ones = np.ones((L,1),dtype=np.float32)
    return np.c_[Ones,matrix] #vertical concatenetion of 2 matrice

def forward(W,X):
    """
    forward calculation 
    y = f(X*W^T) (* is dot calculation)
    f:  activation function
    """
    return activation_func(np.dot(add_bias(X),W))

#------------------------------------------------

N = float(len(T))  #the number of data

print("start learning...")
for i in range(3000):
    #1 sample selection
    index = np.random.randint(N) #select a sample at rondom from dataset
    x = farray([X[index]]) #e.g. x = [[1.0,0.0]]
    t = farray(T[index]) #e.g. t = [[1.0]]
   
    #class prediction (y: response)
    y = forward(W,x) #e.g. y = [[-1.0]]
    #update learning parameter
    if -t[0]*y[0] < 0:  #success case
        pass
    else:               #miss classification case
        W = W - lr * np.dot(add_bias(x).T,y) 
        
        iteration = str(i).zfill(4)
        D = perceptron_criteria(W)
        W_norm = np.linalg.norm(W) # norm calculation
        normalized_W = W/W_norm
        print("iter:{}, D:{:1.2f}, w:{}, w_norm:{:1.2f}, normalized w:{}".format(iteration,D,W.flatten(),W_norm,normalized_W.flatten()))

print("finish")
