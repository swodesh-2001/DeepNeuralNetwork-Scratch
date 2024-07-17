import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import DeepNeuralNetwork

def create_random_dataset(num_samples, input_dim): 
    X = np.linspace(-2 * np.pi, 2 * np.pi, num_samples).reshape(num_samples, input_dim)
    y = np.sin(X)
    X_norm = (X - X.mean()) / X.std()
    return X_norm, y

def plot_predictions(X, y, y_pred):
    plt.figure(figsize=(10, 4))
    plt.plot(y[:, 0], label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle='dashed', color='r')
    plt.title('Actual vs Predicted')
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.legend()
    plt.grid(True)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Neural Network Training')
    parser.add_argument('--num_epochs', type=int, default=  100, help='Number of epochsfor training')
    parser.add_argument('--learning_rate', type=float, default= 0.01, help='Learning rate for training')
    return parser.parse_args()

def main():
    args = parse_arguments()
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    train_x, train_y = create_random_dataset(num_samples= 1000, input_dim = 1) 
    layer_dims = [1, 20, 7, 5, 1] 
    dnn = DeepNeuralNetwork( layer_dims = layer_dims , 
                            hidden_layer_activation = "relu" , 
                            output_layer_activation = "tanh" , 
                            loss_function = "mse") 
    dnn.train(train_x, train_y, num_epochs = num_epochs, learning_rate=learning_rate, print_cost=True) 
    dnn.plot_epochs()  

if __name__ == "__main__":
    main()
