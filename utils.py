import numpy as np

def sigmoid(z):
 
    s = 1 / (1 + np.exp(-z))
    cache = z
    return s,cache

def sigmoid_backward(dA, activation_cache):
 
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu(z):
 
    r = np.maximum(0, z)
    cache = z
    return r,cache

def relu_backward(dA, activation_cache):
 
    Z = activation_cache
    dZ = np.array(dA, copy=True)  
    
   
    dZ[Z <= 0] = 0
    return dZ

 