# ==========================================
# EfficientNet-B0 Product Classifier Evaluation
# ==========================================

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/product_classifier.pt"
IMAGE_SIZE = 96

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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# Load Model
# ==========================================

print("Loading EfficientNet-B0 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.efficientnet_b0(
    weights=None
)

for parameter in model.features.parameters():
    parameter.requires_grad = False

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(
        in_features,
        len(class_names)
    )
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ==========================================
# Preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Test Dataset
# ==========================================
#
# The project already contains five exported
# test images with their labels encoded in
# their filenames.
#
# test_1_Ankle_boot.png
# test_2_Pullover.png
# test_3_Trouser.png
# test_4_Trouser.png
# test_5_Shirt.png
#
# These are sample evaluation images.
# ==========================================

test_images = [
    ("sample_images/test_1_Ankle_boot.png", "Ankle boot"),
    ("sample_images/test_2_Pullover.png", "Pullover"),
    ("sample_images/test_3_Trouser.png", "Trouser"),
    ("sample_images/test_4_Trouser.png", "Trouser"),
    ("sample_images/test_5_Shirt.png", "Shirt")
]


# ==========================================
# Prediction
# ==========================================

y_true = []
y_pred = []

print("\n========== PRODUCT CLASSIFIER EVALUATION ==========")

for image_path, true_class in test_images:

    if not os.path.exists(image_path):
        print("Missing:", image_path)
        continue

    image = Image.open(
        image_path
    ).convert("L")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    predicted_class = class_names[
        predicted_index
    ]

    y_true.append(
        class_names.index(true_class)
    )

    y_pred.append(
        predicted_index
    )

    print("\nImage:", image_path)
    print("Actual     :", true_class)
    print("Predicted  :", predicted_class)
    print(
        "Confidence :",
        round(confidence * 100, 2),
        "%"
    )


# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n========== EVALUATION RESULTS ==========")

print(
    "Test accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ==========================================
# Classification Report
# ==========================================

labels_present = sorted(
    set(y_true) | set(y_pred)
)

target_names_present = [
    class_names[i]
    for i in labels_present
]

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=target_names_present,
        zero_division=0
    )
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=range(len(class_names))
)

print("\n========== CONFUSION MATRIX ==========")

print(cm)


# ==========================================
# Save Confusion Matrix
# ==========================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(cm)

plt.title(
    "EfficientNet-B0 Product Classification Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(class_names)),
    class_names
)

for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

OUTPUT_PATH = (
    "product_classifier_confusion_matrix.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    "\nConfusion matrix saved to:"
)

print(
    OUTPUT_PATH
)