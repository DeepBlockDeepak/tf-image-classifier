import sys
from pathlib import Path

from argparse_config import create_parser

# add the parent directory of the current script to the Python path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_classification.model import (  # import the relevant package
    ImageClassifier,
)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Convert the file paths to Path objects
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)

    # Determine the root directory based on the script location
    root_dir = Path(__file__).resolve().parent.parent

    # Construct the class map path
    class_map_path = root_dir / "scripts" / "imagenet_class_map.txt"

    classifier = ImageClassifier(str(model_path), str(class_map_path))

    """
    Classify the user's image as specified by their supplied file path
    """
    predicted_class, confidence = classifier.classify(str(image_path))

    classifier.print_prediction(
        str(model_path), str(image_path), predicted_class, confidence
    )
