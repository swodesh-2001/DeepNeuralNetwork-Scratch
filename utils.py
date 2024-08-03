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
    r = np.maximum(alpha, z)
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

 
def batch_generator(X, Y, batch_size, shuffle=True):
    m = X.shape[1]
    mini_batches = []

    if shuffle:
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
    else:
        shuffled_X = X
        shuffled_Y = Y

    num_complete_minibatches = m // batch_size
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * batch_size : (k + 1) * batch_size]
        mini_batch_Y = shuffled_Y[:, k * batch_size : (k + 1) * batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * batch_size :]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * batch_size :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches

def plot_actual_vs_predicted(nn, X_norm, y, learning_rate=0.01,batch_size = 4):
    epochs_list = [10, 50, 100, 500, 1000,2000]
    
    plt.figure(figsize=(15, 8))
    
    for i, epochs in enumerate(epochs_list):
        # Train the model
        nn.train(X_norm, y, num_epochs= epochs, learning_rate=learning_rate,batch_size= batch_size,shuffle= False, print_cost = False)
        
        # Generate predictions
        y_pred = nn.predict(X = X_norm ) 
        
        # Plot predictions
        plt.subplot(2, 3, i + 1)
        plt.plot(y.squeeze(), label='Actual')
        plt.plot(y_pred.squeeze(), label='Predicted', linestyle='dashed', color='r')
        plt.title(f'Actual vs Predicted for {epochs} epochs')
        plt.xlabel('X')
        plt.ylabel('Attenuated Sine wave')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()