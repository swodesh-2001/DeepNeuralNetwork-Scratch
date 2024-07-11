import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import DeepNeuralNetwork

def create_random_dataset(num_samples, input_dim):
    np.random.seed(1)
    X = np.random.randn(input_dim, num_samples)
    Y = np.zeros((1, num_samples))
 
    for i in range(num_samples):
        if np.sum(X[:, i]) > 0:
            Y[0, i] = 1
        else:
            Y[0, i] = 0

    return X, Y.astype(int)

def plot_costs(costs, title):
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title(title)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Neural Network Training')
    parser.add_argument('--num_iterations', type=int, default=2500, help='Number of iterations for training')
    parser.add_argument('--learning_rate', type=float, default=1, help='Learning rate for training')
    return parser.parse_args()

def main():
    args = parse_arguments()
    num_iterations = args.num_iterations
    learning_rate = args.learning_rate
    train_x, train_y = create_random_dataset(1000, 12288)
    val_x, val_y = create_random_dataset(200, 12288) 
    layer_dims = [12288, 20, 7, 5, 1] 
    dnn = DeepNeuralNetwork(layer_dims) 
    parameters, costs = dnn.train(train_x, train_y, num_iterations=num_iterations, learning_rate=learning_rate, print_cost=True) 
    plot_costs(costs, "Training Cost") 
    AL_val, _ = dnn.L_model_forward(val_x)
    val_cost = dnn.compute_cost(AL_val, val_y)
    print(f"Validation Cost: {val_cost}")

if __name__ == "__main__":
    main()
