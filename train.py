import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import DeepNeuralNetwork

def create_random_dataset(num_samples): 
    X = np.linspace(0, 10 * np.pi, num_samples).reshape(1, num_samples)
    y = 10 * np.sin(X) * np.exp(-0.3 * X) 
    return X, y

def plot_predictions(y, y_pred):
    plt.figure(figsize=(10, 4))
    plt.plot(y, label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle='dashed', color='r')
    plt.title('Actual vs Predicted')
    plt.xlabel('X')
    plt.ylabel('sin(X)')
    plt.legend()
    plt.grid(True)
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Neural Network Training')
    parser.add_argument('--num_epochs', type=int, default= 1000, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default= 16, help='Batch size for training')
    return parser.parse_args()

def main():
    args = parse_arguments()
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    train_x, train_y = create_random_dataset(num_samples= 1000)
    layer_dims = [1, 16,32,16, 1]
    dnn = DeepNeuralNetwork(layer_dims=layer_dims, hidden_layer_activation="relu", output_layer_activation="linear", loss_function="mse")
    dnn.train(train_x, train_y, num_epochs=num_epochs, learning_rate=learning_rate,batch_size= batch_size,shuffle= False, print_cost=True)
    dnn.plot_epochs()
    y_pred = dnn.predict(X = train_x ) 
    plot_predictions(y= train_y.squeeze() , y_pred = y_pred.squeeze())


if __name__ == "__main__":
    main()
