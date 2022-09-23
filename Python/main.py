import math
import numpy as np
import data_loader as dl

learning_rate = 0
number_of_epochs = 0
batch_size = 0


class Model:

    def __init__(self):
        self.w1 = np.random.normal(0, 1, (16, 784))
        self.w2 = np.random.normal(0, 1, (16, 16))
        self.w3 = np.random.normal(0, 1, (10, 16))
        self.b1 = np.zeros((16, 1))
        self.b2 = np.zeros((16, 1))
        self.b3 = np.zeros((10, 1))

    @staticmethod
    def relu(x):
        zero = np.zeros(x.shape)
        return np.maximum(zero, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(int)

    def forward_prop(self, inputs):
        cache = {'z': [], 'a': []}
        inputs /= 255

        z1 = (self.w1 @ inputs) + self.b1
        a1 = self.relu(z1)
        cache['z'].append(z1)
        cache['a'].append(a1)

        z2 = (self.w2 @ a1) + self.b2
        a2 = self.relu(z2)
        cache['z'].append(z2)
        cache['a'].append(a2)

        z3 = (self.w3 @ a2) + self.b3
        a3 = self.relu(z3)
        cache['z'].append(z3)
        cache['a'].append(a3)

        return cache

    def predict(self, inputs):
        cache = self.forward_prop(inputs)
        y = cache['a'][2]
        rows, columns = y.shape
        for i in range(columns):
            biggest = -np.inf
            index = -1
            for j in range(rows):
                if y[j][i] >= biggest:
                    biggest = y[j][i]
                    index = j
            for j in range(rows):
                y[j][i] = 0
            y[index][i] = 1
        return y

    def loss(self, inputs, labels):
        cache = self.forward_prop(inputs)
        y = cache['a'][2]
        loss = y - labels
        loss = np.power(loss, 2)
        loss = np.sum(loss, axis=0, keepdims=True)
        loss = np.sum(loss, axis=1, keepdims=True)
        loss /= len(inputs)
        return loss

    def back_prop(self, inputs, labels, cache):
        m = len(inputs)
        grads = {'dw': [], 'db': []}

        da3 = 2 * (cache['a'][2] - labels)
        db3 = da3 * self.relu_derivative(cache['z'][2])
        dw3 = db3 @ cache['a'][1].transpose()

        da2 = self.w3.transpose() @ (self.relu_derivative(cache['z'][2]) * da3)
        db2 = da2 * self.relu_derivative(cache['z'][1])
        dw2 = db2 @ cache['a'][0].transpose()

        da1 = self.w2.transpose() @ (self.relu_derivative(cache['z'][1]) * da2)
        db1 = da1 * self.relu_derivative(cache['z'][0])
        dw1 = db1 @ inputs.transpose()

        db3 = np.sum(db3, axis=1, keepdims=True) / m
        db2 = np.sum(db2, axis=1, keepdims=True) / m
        db1 = np.sum(db1, axis=1, keepdims=True) / m
        dw3 /= m
        dw2 /= m
        dw1 /= m

        grads['db'].append(db1)
        grads['db'].append(db2)
        grads['db'].append(db3)
        grads['dw'].append(dw1)
        grads['dw'].append(dw2)
        grads['dw'].append(dw3)
        return grads

    def optimizer_step(self, inputs, labels, learning_rate):
        cache = self.forward_prop(inputs)
        grads = self.back_prop(inputs, labels, cache)
        self.w1 -= learning_rate * grads['dw'][0]
        self.w2 -= learning_rate * grads['dw'][1]
        self.w3 -= learning_rate * grads['dw'][2]
        self.b1 -= learning_rate * grads['db'][0]
        self.b2 -= learning_rate * grads['db'][1]
        self.b3 -= learning_rate * grads['db'][2]

    def sgd(self, inputs, labels, learning_rate, number_of_epochs, batch_size):
        m = len(inputs)
        for i in range(number_of_epochs):
            np.random.shuffle(inputs)
            number_of_batches = math.floor(m / batch_size)
            for j in range(number_of_batches):
                x = inputs[j * batch_size: (j + 1) * batch_size]
                y = labels[j * batch_size: (j + 1) * batch_size]
                self.optimizer_step(x.transpose(), y.transpose(), learning_rate)
                print("loss: ", self.loss(x.transpose(), y.transpose())[0][0])
            if number_of_batches * batch_size < m:
                x = inputs[number_of_batches * batch_size:]
                y = labels[number_of_batches * batch_size:]
                self.optimizer_step(x.transpose(), y.transpose(), learning_rate)


if __name__ == '__main__':

    dl.fetch_data()
    train_images, train_labels = dl.smooth_data(dl.train_set)
    model = Model()
    model.sgd(train_images, train_labels, learning_rate, number_of_epochs, batch_size)

    test_images, test_labels = dl.smooth_data(dl.test_set)
    predictions = model.predict(test_images.transpose())
    predictions = predictions.transpose()

    test_items = len(test_labels)
    true_answers = 0.0
    for i in range(test_items):
        if (test_labels[i] == predictions[i]).all():
            true_answers += 1
    percentage = (true_answers / test_items) * 100
    print("percentage: ", percentage)
