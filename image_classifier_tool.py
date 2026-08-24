# ==========================================
# EfficientNet-B0 Image Classifier Tool
# ==========================================

import os
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms, models


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

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model = models.efficientnet_b0(
        weights=None
    )

    # Freeze feature extractor
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

    return model


# ==========================================
# Image Preprocessing
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
# Prediction Function
# ==========================================

def classify_image(image_path):

    print("\n========== IMAGE CLASSIFIER TOOL ==========")

    print("Image:", image_path)

    model = load_model()

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

    result = {
        "prediction": predicted_class,
        "confidence": confidence,
        "confidence_percent": round(
            confidence * 100,
            2
        ),
        "class_index": predicted_index,
        "model": "EfficientNet-B0"
    }

    print(
        "Prediction :",
        predicted_class
    )

    print(
        "Confidence :",
        round(confidence * 100, 2),
        "%"
    )

    print(
        "Class index:",
        predicted_index
    )

    return result


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    image_path = (
        "sample_images/test_1_Ankle_boot.png"
    )

    result = classify_image(
        image_path
    )

    print("\n========== RESULT ==========")

    print(result)