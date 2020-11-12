import os
import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    # Check raw data directory
    raw_data_dir = "../data-python"
    if not os.path.exists(raw_data_dir):
        print("Raw data doesn't exist.")
        exit(0)
    # Create data directory
    data_dir = "../data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # Train data preparation
    train_data = np.zeros((50000, 32, 32, 3), dtype=np.float32)
    train_label = np.zeros((50000, 10))
    raw_data = np.zeros((50000, 32 * 32 * 3))
    for i in range(1, 6):
        data = unpickle(os.path.join(raw_data_dir, "data_batch_{}".format(i)))
        raw_data[(i - 1) * 10000:i * 10000] = data[b'data']
        for j in range(10000):
            train_label[(i - 1) * 10000 + j, data[b'labels'][j]] = 1

    raw_data = raw_data.reshape((-1, 3, 32, 32))
    for i in range(3):
        train_data[:, :, :, i] = raw_data[:, i, :, :] / 255

    np.save(os.path.join(data_dir, "train_data.npy"), train_data)
    np.save(os.path.join(data_dir, "train_label.npy"), train_label)

    # Test data preparation
    test_data = np.zeros((10000, 32, 32, 3), dtype=np.float32)
    data = unpickle(os.path.join(raw_data_dir, "test_batch"))
    raw_data = data[b'data']
    raw_data = raw_data.reshape(-1, 3, 32, 32)
    for i in range(3):
        test_data[:, :, :, i] = raw_data[:, i, :, :] / 255

    test_label = np.zeros((10000, 10))
    for i in range(10000):
        test_label[i, data[b'labels'][i]] = 1
    np.save(os.path.join(data_dir, "test_label.npy"), test_label)
    np.save(os.path.join(data_dir, "test_data.npy"), test_data)