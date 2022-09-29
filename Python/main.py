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
        predicted_value = np.argmax(y, axis=0)
        return predicted_value

    def loss(self, inputs, labels):
        cache = self.forward_prop(inputs)
        y = cache['a'][2]
        loss = y - labels
        loss = np.power(loss, 2)
        loss = np.sum(loss, axis=0, keepdims=True)
        loss = np.sum(loss, axis=1, keepdims=True)
        loss /= len(inputs)
        return loss[0][0]

    def back_prop(self, inputs, labels, cache):
        m = len(inputs)
        grads = {'dw': [], 'db': []}

        da3 = 2 * (cache['a'][2] - labels)
        db3 = da3 * self.relu_derivative(cache['z'][2])
        dw3 = (db3 @ cache['a'][1].transpose()) / m

        da2 = self.w3.transpose() @ (self.relu_derivative(cache['z'][2]) * da3)
        db2 = da2 * self.relu_derivative(cache['z'][1])
        dw2 = (db2 @ cache['a'][0].transpose()) / m

        da1 = self.w2.transpose() @ (self.relu_derivative(cache['z'][1]) * da2)
        db1 = da1 * self.relu_derivative(cache['z'][0])
        dw1 = (db1 @ inputs.transpose()) / m

        db3 = np.sum(db3, axis=1, keepdims=True) / m
        db2 = np.sum(db2, axis=1, keepdims=True) / m
        db1 = np.sum(db1, axis=1, keepdims=True) / m

        grads['db'].append(db1)
        grads['db'].append(db2)
        grads['db'].append(db3)
        grads['dw'].append(dw1)
        grads['dw'].append(dw2)
        grads['dw'].append(dw3)
        return grads

    def optimizer_step(self, inputs, labels):
        cache = self.forward_prop(inputs)
        grads = self.back_prop(inputs, labels, cache)
        self.w1 -= learning_rate * grads['dw'][0]
        self.w2 -= learning_rate * grads['dw'][1]
        self.w3 -= learning_rate * grads['dw'][2]
        self.b1 -= learning_rate * grads['db'][0]
        self.b2 -= learning_rate * grads['db'][1]
        self.b3 -= learning_rate * grads['db'][2]

    def sgd(self, inputs, labels):
        number_of_batches = math.floor(len(inputs) / batch_size)
        for i in range(number_of_epochs):
            np.random.shuffle(inputs)
            for j in range(number_of_batches):
                x = inputs[j * batch_size: (j + 1) * batch_size]
                y = labels[j * batch_size: (j + 1) * batch_size]
                x = x.transpose()
                y = y.transpose()
                self.optimizer_step(x, y)
                print("loss: ", self.loss(x, y))


if __name__ == '__main__':

    dl.fetch_data()
    train_images, train_labels = dl.smooth_data(dl.train_set)

    model = Model()
    model.sgd(train_images, train_labels)

    test_images, test_labels = dl.smooth_data(dl.test_set)
    predictions = model.predict(test_images.transpose())
    test_values = np.argmax(test_labels, axis=1)
    equality = np.equal(predictions, test_values).astype(int)
    true_answers = np.sum(equality)
    accuracy = (true_answers / len(test_labels)) * 100
    print("accuracy: ", accuracy)
