# argparse_config.py
import argparse
from pathlib import Path


def create_parser():
    parser = argparse.ArgumentParser(description="Classify an image.")
    parser.add_argument("model_path", type=Path, help="Path to the classifer model.")
    parser.add_argument(
        "image_path", type=Path, help="Path to the image to be classified."
    )
    return parser
