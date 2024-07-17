import numpy as np
from utils import *

class DeepNeuralNetwork:
    def __init__(self, layer_dims, hidden_layer_activation, output_layer_activation, loss_function):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters(layer_dims)
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        self.loss_function = loss_function
        self.costs = []

    def initialize_parameters(self, layer_dims):
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) * 0.1

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
        elif activation == "leaky_relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = leaky_relu(Z)
        elif activation == "tanh":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z)
        elif activation =="linear" :
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A,activation_cache = Z,Z
        else:
            raise ValueError("Invalid activation function. Supported activations: 'sigmoid', 'relu', 'leaky_relu', 'tanh'")

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation=self.hidden_layer_activation)
            caches.append(cache)

        AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation=self.output_layer_activation)
        caches.append(cache)

        assert(AL.shape == (1, X.shape[1]))

        return AL, caches

    def compute_cost(self, AL, Y, loss_type='cross_entropy'):
        m = Y.shape[1]
        
        if loss_type == 'cross_entropy':
            cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
        elif loss_type == 'mse':
            cost = (1/m) * np.sum((AL - Y) ** 2)
        elif loss_type == 'mae':
            cost = (1/m) * np.sum(np.abs(AL - Y))
        else:
            raise ValueError("Invalid loss type. Supported types: 'cross_entropy', 'mse', 'mae'")
        
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1./m) * np.dot(dZ, A_prev.T)
        db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        elif activation == "leaky_relu":
            dZ = leaky_relu_backward(dA, activation_cache)
        elif activation == "tanh":
            dZ = tanh_backward(dA, activation_cache)
        elif activation == "linear":
            dZ = dA
        else:
            raise ValueError("Invalid activation function. Supported activations: 'sigmoid', 'relu', 'leaky_relu', 'tanh'")

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        if self.loss_function == 'cross_entropy':
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        elif self.loss_function == 'mse':
            dAL = 2 * (AL - Y) / m
        elif self.loss_function == 'mae':
            dAL = np.where(AL > Y, 1, -1) / m

        current_cache = caches[-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation=self.output_layer_activation)

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation=self.hidden_layer_activation)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

        return self.parameters

    def train(self, X, Y, learning_rate=0.0075, num_epochs=100, batch_size = 64, shuffle = True ,epoch_verbose = 10 , print_cost=False):
        self.costs = []

        for i in range(num_epochs):
            minibatches = batch_generator(X , Y , batch_size = batch_size, shuffle = shuffle)
            minibatch_num = len(minibatches)
            cost = 0
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                AL, caches = self.L_model_forward(minibatch_X)
                cost += self.compute_cost(AL, minibatch_Y, loss_type=self.loss_function)
                grads = self.L_model_backward(AL, minibatch_Y, caches)
                self.parameters = self.update_parameters(grads, learning_rate)
            cost = cost/minibatch_num
            if print_cost and i % epoch_verbose == 0:
                print(f"Cost after iteration {i}: {cost}")
            
            self.costs.append(cost)
    
    def plot_epochs(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations (per hundreds)')
        plt.title("Epochs vs Cost")
        plt.show()

    def predict(self, X): 
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation=self.hidden_layer_activation)
        AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation=self.output_layer_activation)

        return AL
