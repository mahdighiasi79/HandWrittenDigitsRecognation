import numpy as np
import pickle

train_set = []
test_set = []


def fetch_data():
    # Reading The Train Set
    train_images_file = open('..\\Datasets\\train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('..\\Datasets\\train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))

    # Reading The Test Set
    test_images_file = open('..\\Datasets\\t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('..\\Datasets\\t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))


def smooth_data(dataset):
    inputs = []
    record = []

    for i in range(len(dataset)):
        for j in range(len(dataset[i][0])):
            record.append(dataset[i][0][j][0])
        inputs.append(record)
        record = []

    labels = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i][1])):
            record.append(dataset[i][1][j][0])
        labels.append(record)
        record = []

    inputs = np.asarray(inputs)
    labels = np.asarray(labels)
    return inputs, labels


class Data:
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels


if __name__ == '__main__':
    fetch_data()
    inputs, labels = smooth_data(train_set)
    data = Data(inputs, labels)
    with open('..\\Datasets\\train_set', 'wb') as train_set:
        pickle.dump(data, train_set)
    inputs, labels = smooth_data(test_set)
    data = Data(inputs, labels)
    with open('..\\Datasets\\test_set', 'wb') as test_set:
        pickle.dump(data, test_set)
