import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import DeepNeuralNetwork
from utils import plot_actual_vs_predicted

def create_random_dataset(num_samples): 
    X = np.linspace(-2 * np.pi, 2 * np.pi, num_samples).reshape(1, num_samples)
    y = (0.25 * np.sin(X) + 0.25) * np.exp(-0.1 * X) 
    X_norm = (X - X.mean()) / X.std()
    return X_norm, y 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Neural Network Training')
    parser.add_argument('--num_epochs', type=int, default= 1000, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default= 4, help='Batch size for training')
    return parser.parse_args()

def main():
    args = parse_arguments()
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    train_x, train_y = create_random_dataset(num_samples= 500)
    layer_dims = [1,20,20,1]
    dnn = DeepNeuralNetwork(layer_dims=layer_dims, hidden_layer_activation="relu", output_layer_activation="sigmoid", loss_function="mse")
    # dnn.train(train_x, train_y, num_epochs=num_epochs, learning_rate=learning_rate,batch_size= batch_size,shuffle= False, print_cost=True)
    # dnn.plot_epochs()
    # y_pred = dnn.predict(X = train_x ) 

    plot_actual_vs_predicted(dnn, train_x, train_y , learning_rate= learning_rate , batch_size= batch_size)
    dnn.plot_epochs()


if __name__ == "__main__":
    main()
