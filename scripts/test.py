import sys
from pathlib import Path

from argparse_config import create_parser

# add the parent directory of the current script to the Python path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_classification.model import (  # import the relevant package
    ImageClassifier,
)

classifier = ImageClassifier("models/model_b3", "scripts/imagenet_class_map.txt")

print(
    type(classifier.class_map),
    type(list(classifier.class_map.keys())[0]),
    type(list(classifier.class_map.values())[0]),
)
