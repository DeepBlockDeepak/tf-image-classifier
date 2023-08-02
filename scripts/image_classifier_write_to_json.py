import datetime
import json
import os
import sys
from pathlib import Path

from argparse_config import create_parser

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.image_classification.model import (
    ImageClassifier,  # import the relevant package
)


def generate_results_dict(model_str, image_path, predicted_classes, confidences):
    model_name = "EfficientNet" + os.path.basename(model_str).split("_")[1].upper()
    image_name = os.path.basename(image_path).split(".")[0]

    # Create a dictionary to store the results
    results = {
        "Model Used": model_name,
        "Image Classified": image_name,
        "Predictions": [
            {"class": pred_class, "confidence": round(float(conf), 2)}
            for pred_class, conf in zip(predicted_classes, confidences)
        ],
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return results


def append_results_to_data(output_file, results):
    if os.path.exists(output_file):
        # If the file already exists, load the existing data and append the new results
        with open(output_file, "r") as f:
            data = json.load(f)
        data.append(results)
    else:
        # If the file doesn't exist, start a new list of results
        data = [results]

    return data


def write_data_to_json(output_file, data):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the data to the output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def write_top_predictions_to_json(
    model_str, image_path, predicted_classes, confidences
):
    results = generate_results_dict(
        model_str, image_path, predicted_classes, confidences
    )
    output_file = (
        f"output/{os.path.basename(image_path).split('.')[0]}_predictions.json"
    )
    data = append_results_to_data(output_file, results)

    write_data_to_json(output_file, data)


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
    Classify the image and write the top 5 predictions to a JSON file in /output
    """
    predicted_classes, confidences = classifier.classify_top_k(str(image_path))
    write_top_predictions_to_json(
        str(model_path), str(image_path), predicted_classes, confidences
    )
