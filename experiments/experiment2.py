import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.image_classification.model import ImageClassifier


# hard-coding the true data to the true results as found in the imagenet class_map
def load_your_test_data():
    data = [
        ("stella_boston_terrier.jpg", "Boston bull, Boston terrier"),
        ("macaque.jpeg", "macaque"),
    ]

    process_image_path = lambda x: ("images/" + x[0], x[1])

    data = list(map(process_image_path, data))

    return data


# Initialize a new wandb run
wandb.init(
    project="tf-image-classifier",
    config={
        "architecture": "EfficientNet",
        "dataset": "ImageNet",
    },
)

# Initialize your model
classifier = ImageClassifier(
    model_path="models/model_b1", class_map_path="scripts/imagenet_class_map.txt"
)

# Load your test data
# This should be a list of (image_path, true_label) pairs
test_data = load_your_test_data()

# Evaluate your model on the test set
# would need to be someting like : [(),()]
for image_path, true_label in test_data:
    # predicted_class is the string value of the class_map dict
    predicted_class, confidence = classifier.classify(image_path)

    # Compute whether the prediction was correct
    correct = predicted_class == true_label

    # Log the result with wandb
    wandb.log({"correct": correct, "confidence": confidence})

# Finish the wandb run
wandb.finish()
