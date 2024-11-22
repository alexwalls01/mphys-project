import pickle
import numpy as np

def one_hot(x, k, dtype=np.float32):
  return np.array(x[:, None] == np.arange(k), dtype)

def load_CIFAR_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    images = np.array(data[b'data'])
    labels = one_hot(np.array(data[b'labels']), 10)
    return images, labels

def load_CIFAR_annotations(file):
    annotations = np.load(file)
    return annotations