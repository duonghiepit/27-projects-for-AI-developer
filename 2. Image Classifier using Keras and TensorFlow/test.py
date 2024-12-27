import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from dataset_loader import DatasetLoader

# Load the model from the file
loaded_model = load_model('./model/mnist_cnn_model.h5')

# Khởi tạo lớp DatasetLoader và tải dữ liệu
dataset_loader = DatasetLoader()
dataset_loader.load_data()
test_images, test_labels = dataset_loader.get_test_data()

# Make predictions on test images
predictions = loaded_model.predict(test_images)
print(f"Prediction for first test image: {np.argmax(predictions[0])}")

plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted label: {np.argmax(predictions[0])}")
plt.show()