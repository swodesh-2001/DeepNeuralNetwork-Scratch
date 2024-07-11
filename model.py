import numpy as np
from utils import *

class DeepNeuralNetwork:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters(layer_dims)
        
    def initialize_parameters(self, layer_dims):
        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation="sigmoid")
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation="sigmoid")
        caches.append(cache)

        assert(AL.shape == (1, X.shape[1]))

        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation="sigmoid")

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return self.parameters

    def train(self, X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        costs = []

        for i in range(num_iterations):
            AL, caches = self.L_model_forward(X)
            cost = self.compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            self.parameters = self.update_parameters(grads, learning_rate)

            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
            if i % 100 == 0:
                costs.append(cost)

        return self.parameters, costs
