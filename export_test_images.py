import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================
# Load Fashion-MNIST test data
# ==========================================

print("Loading Fashion-MNIST test data...")

(_, _), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)


# ==========================================
# Class names
# ==========================================

class_names = [
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_boot"
]


# ==========================================
# Create output folder
# ==========================================

os.makedirs(
    "sample_images",
    exist_ok=True
)


# ==========================================
# Export 5 test images
# ==========================================

print("\n========== EXPORTING TEST IMAGES ==========")

for i in range(5):

    image = x_test[i]
    label = y_test[i]

    filename = (
        f"sample_images/"
        f"test_{i+1}_{class_names[label]}.png"
    )

    Image.fromarray(image).save(filename)

    print(
        f"Saved: {filename} "
        f"| True label: {class_names[label]}"
    )


print("\n5 sample images exported successfully.")