import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets

class DatasetLoader:
    def __init__(self, dataset='mnist'):
        self.dataset = dataset
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def load_data(self):
        # Tải bộ dữ liệu MNIST
        if self.dataset == 'mnist':
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.mnist.load_data()
        else:
            raise ValueError(f"Dataset {self.dataset} is not supported yet")

        # Tiền xử lý dữ liệu
        self._preprocess_data()

    def _preprocess_data(self):
        # Chuẩn hóa giá trị pixel từ 0 đến 1
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

        # Reshape dữ liệu về kích thước (28, 28, 1)
        self.train_images = self.train_images.reshape((self.train_images.shape[0], 28, 28, 1))
        self.test_images = self.test_images.reshape((self.test_images.shape[0], 28, 28, 1))

        # Chuyển labels sang dạng one-hot encoding
        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)

    def get_train_data(self):
        return self.train_images, self.train_labels

    def get_test_data(self):
        return self.test_images, self.test_labels
