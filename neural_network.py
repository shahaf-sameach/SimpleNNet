import numpy as np
import time
from activation import sigmoid, dsigmoid, relu ,drelu, softmax
from sklearn.utils import shuffle

class NeuralNetwork(object):
    def __init__(self, input, hidden, output):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """

        # for replicability 
        np.random.seed(1)

        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output
        # set up array of 1s for activations
        self.ai = np.array([1.0] * self.input)
        self.ah = np.array([1.0] * self.hidden)
        self.ao = np.array([1.0] * self.output)
        # create randomized weights
    
        self.wi = np.random.randn(self.input, self.hidden) 
        self.wo = np.random.randn(self.hidden, self.output) 
        
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs')
        
        # input activations
        self.ai = np.append(inputs, [1]) # add bias node
        
        # hidden activations
        self.ah = sigmoid(self.ai.dot(self.wi))
        # self.ah = relu(self.ai.dot(self.wi))
        
        # output activations
        self.ao = sigmoid(self.ah.dot(self.wo))
        # self.ao = relu(self.ah.dot(self.wo))
        
        return softmax(self.ao)

    def backPropagate(self, targets, N):
        """
        :param targets: y values
        :param N: learning rate
        :return: updated weights and current error
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets')
        
        # calculate error
        error = -(np.array(targets) - self.ao)
        output_deltas = dsigmoid(self.ao) * error
        
        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        error = output_deltas.dot(self.wo.T)
        hidden_deltas = dsigmoid(self.ah) * error
        
        # update the weights connecting hidden to output
        change = np.outer(self.ah, output_deltas)
        self.wo -= N * change + self.co
        self.co = change
        
        # update the weights connecting input to hidden
        change = np.outer(self.ai, hidden_deltas)
        self.wi -= N * change + self.ci
        self.ci = change
        
        error = sum(0.5 * (targets - self.ao) ** 2)
        return error

    def train(self, x_train_set, y_train_set, epoch = 100, N = 0.0002, x_test=None, y_test=None):
        # N: learning rate

        for i in range(epoch):
            error = 0.0
            t0 = time.time()
            x_train, y_train = shuffle(x_train_set, y_train_set, random_state=0)
            for x, y in zip(x_train, y_train):
                inputs = x
                targets = y
                self.feedForward(inputs)
                error = self.backPropagate(targets, N)
            
            if i % 1 == 0:
                train_acc = self.evaluate(x_train, y_train)
                if x_test is None or y_test is None:
                    print('epoch %s: train_acc %-.5f took %s sec' % (i, train_acc, int(time.time() - t0)))
                else:
                    test_acc = self.evaluate(x_test, y_test)
                    print('epoch %s: train_acc %-.5f test_acc %-.5f took %s sec' % (i, train_acc, test_acc, int(time.time() - t0)))

    def predict(self, X):
        predictions = []
        for p in X:
            predictions.append(np.argmax(self.feedForward(p)))
        return predictions

    def evaluate(self, x_test, y_test):
        test_results = [(np.argmax(self.feedForward(x)), np.argmax(y))
                        for (x, y) in zip(x_test, y_test)]
                              
        return float(sum(int(x == y) for (x, y) in test_results))/float(len(test_results))

    def save(self, file_name="weights.csv"):
        weights_output = np.append(self.wi, self.wo)
        np.savetxt(file_name,weights_output,fmt='%.4f',delimiter=',')
        print("saved weights in {}".format(file_name))

    def load(self, file_name="weights.csv"):
        ws = np.loadtxt(file_name, delimiter=',')
        wi_flat = ws[:self.wi.shape[0] * self.wi.shape[1]]
        wo_flat = ws[self.wi.shape[0] * self.wi.shape[1]:]
        self.wi = wi_flat.reshape((self.wi.shape[0],self.wi.shape[1]))
        self.wo = wo_flat.reshape((self.wo.shape[0],self.wo.shape[1]))
        print("loaded weights from {}".format(file_name))


