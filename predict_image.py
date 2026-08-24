import tensorflow as tf
import numpy as np
from PIL import Image
import os


# ==========================================
# Load saved model
# ==========================================

MODEL_PATH = "fashion_mnist_mobilenetv2.keras"

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ==========================================
# Class names
# ==========================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ==========================================
# Image preprocessing function
# ==========================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("L")

    image = np.array(image)

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Add grayscale channel
    image = np.expand_dims(
        image,
        axis=-1
    )

    # Convert grayscale → RGB
    image = np.repeat(
        image,
        3,
        axis=-1
    )

    # Resize to MobileNetV2 input size
    image = tf.image.resize(
        image,
        (96, 96)
    )

    # Add batch dimension
    image = np.expand_dims(
        image.numpy(),
        axis=0
    )

    return image


# ==========================================
# Predict all sample images
# ==========================================

sample_folder = "sample_images"

print("\n========== ALL SAMPLE PREDICTIONS ==========")

for filename in sorted(os.listdir(sample_folder)):

    if not filename.lower().endswith(".png"):
        continue

    image_path = os.path.join(
        sample_folder,
        filename
    )

    # Preprocess image
    processed_image = preprocess_image(
        image_path
    )

    # Model prediction
    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    # Get predicted class
    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = class_names[
        predicted_index
    ]

    # Get confidence
    confidence = probabilities[
        predicted_index
    ]

    # Extract actual class from filename
    actual_class = filename.split("_", 2)[2]

    actual_class = actual_class.replace(
        ".png",
        ""
    ).replace(
        "_",
        " "
    )

    # Display result
    print("\nImage      :", filename)
    print("Actual     :", actual_class)
    print("Predicted  :", predicted_class)
    print(
        "Confidence :",
        round(confidence * 100, 2),
        "%"
    )

    # Check prediction
    if actual_class.lower() == predicted_class.lower():
        print("Result     : CORRECT")
    else:
        print("Result     : INCORRECT")


print("\n========== PREDICTION COMPLETE ==========")