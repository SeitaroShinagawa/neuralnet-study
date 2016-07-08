#!/usr/bin/python 
import numpy as np
import copy

#np.array(float32) function
def farray(matrix):
    return np.array(matrix,np.float32)

X=farray([[0,0],
          [0,1],
          [1,0],
          [1,1]])
Y=farray([[0],
          [1],
          [1],
          [0]])

W1=np.random.rand(3,2)
W2=np.random.rand(3,1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def add_bias(matrix):
    L=len(matrix)
    Ones=np.ones((L,1),dtype=np.float32)
    return np.c_[Ones,matrix]

def forward_one(W,X):
    return sigmoid(np.dot(add_bias(X),W))

def square_error(y,t):
    return np.dot((y-t).T,y-t)

#def cross_entropy(y,t):
#    return -np.sum(t*np.log(y)+(1-t)*np.log(1-y))
    
lr = 0.1
cost_list=[]
acc_list=[]

for i in range(10000):
    h = forward_one(W1,X)
    y = forward_one(W2,h)
    objective = 0.5*square_error(y,Y)/len(y)
    #print("objective func:",objective)
    err = y-Y
    err1=np.dot(err,W2.T)
    delta_w2=np.dot(add_bias(h).T,err)/4
    delta_w1=np.dot(add_bias(X).T,h*(1-h)*err1[:,1:])/4
    W1 -= lr*delta_w1
    W2 -= lr*delta_w2

    if i%1000 == 0:
        print("gradient checking")
        diff = 10e-5 #small perturbation change
        #check W2 gradient
        pW2=copy.deepcopy(W2)
        nW2=copy.deepcopy(W2)
        #check W2[0][0] gradient
        pW2[0][0] += diff
        nW2[0][0] -= diff
        h = forward_one(W1,X)
        py = forward_one(pW2,h)
        pobj = square_error(py,Y)
        ny = forward_one(nW2,h)
        nobj = square_error(ny,Y)
        errW2 = pobj - nobj - 2*diff*delta_w2[0][0]
        print("errW2:{},pobj:{},nobj:{}".format(errW2,pobj,nobj)) #if errW2>10^-4 then doesn't work well
        print("testing")
        h = forward_one(W1,X)
        y = forward_one(W2,h)
        objective = square_error(y,Y)/len(y)

        accuracy=0
        for i,j in zip(y,Y):
            predict = (np.sign(i-0.5)+1)*0.5
            if predict==j:
                accuracy+=1
        print("accuracy:",accuracy/len(y))
        acc_list.append(accuracy/len(y))

print("finish.")
