import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.image_classification.model import ImageClassifier

# Initialize a new wandb run
wandb.init(
    project="tf-image-classifier",
    config={
        "learning_rate": 0.02,
        "architecture": "EfficientNet",
        "dataset": "ImageNet",
        "epochs": 10,
    },
)

# Initialize your model
classifier = ImageClassifier(
    model_path="models/model_b1", class_map_path="scripts/imagenet_class_map.txt"
)

# Load your training and validation data
# You'll need to replace these lines with your actual data loading code
train_data = load_your_train_data()
validation_data = load_your_validation_data()

# Train the model for the specified number of epochs
for epoch in range(wandb.config["epochs"]):
    # Train your model for one epoch here
    classifier.train(train_data, learning_rate=wandb.config["learning_rate"])

    # Evaluate your model on the validation set here
    metrics = classifier.evaluate(validation_data)

    # Log metrics with wandb
    wandb.log(metrics)

# Finish the wandb run
wandb.finish()
