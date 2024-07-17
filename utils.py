import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    cache = z
    return s, cache

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu(z):
    r = np.maximum(0, z)
    cache = z
    return r, cache

def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def leaky_relu(z, alpha=0.01):
    r = np.where(z > 0, z, alpha * z)
    cache = z
    return r, cache

def leaky_relu_backward(dA, activation_cache, alpha=0.01):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = alpha
    return dZ

def tanh(z):
    t = np.tanh(z)
    cache = z
    return t, cache

def tanh_backward(dA, activation_cache):
    Z = activation_cache
    t = np.tanh(Z)
    dZ = dA * (1 - t ** 2)
    return dZ

 
