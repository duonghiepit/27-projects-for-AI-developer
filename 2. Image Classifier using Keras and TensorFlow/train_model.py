import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from dataset_loader import DatasetLoader  

# Khởi tạo lớp DatasetLoader và tải dữ liệu
dataset_loader = DatasetLoader()
dataset_loader.load_data()

# Lấy dữ liệu huấn luyện và kiểm tra
train_images, train_labels = dataset_loader.get_train_data()
test_images, test_labels = dataset_loader.get_test_data()

# Xây dựng mô hình CNN
model = models.Sequential()

# First Convolution Layer
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

# Second Convolution Layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Third Convolution Layer
model.add(layers.Conv2D(63, (3,3), activation='relu'))

# Flatten the 3D output to 1D and add a Dense layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Output layer with 10 neurons (for 10 digit classes)
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_acc*100:.2f}%")

# Save the model to a file
model.save('./model/mnist_cnn_model.h5')
