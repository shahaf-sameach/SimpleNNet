import numpy as np
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork

if __name__ == '__main__':
    print("loading data...")
    X_data = np.loadtxt('digits_train.csv', delimiter=',')
    y_data = np.loadtxt('digits_train_key.csv', delimiter=',')
    X_data = X_data.astype('float32')
    y_data = y_data.astype('float32')
    X_data /= 255

    Y_data = np.zeros((len(y_data), 10))
    for i in range(len(y_data)):
        Y_data[i, int(y_data[i])] = 1.

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1)

    net = NeuralNetwork(784, 100, 10)

    print("training...")
    net.train(X_train, Y_train, epoch=300, x_test=X_test, y_test=Y_test)
    score = net.evaluate(X_test, Y_test)
    print("acc=%-.3f" % (score * 100))
    net.save(file_name="weights.csv")

    # loading weights
    net.load(file_name="weights.csv")
    # running in prediction mode
    predictions = net.predict(X_data)
    # writing to file
    write_to_file(predictions)

    print("done")




