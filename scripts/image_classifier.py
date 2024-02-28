import base64
import sys
from pathlib import Path

from taipy.gui import Gui

# add the parent directory of the current script to the Python path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_classification.model import (  # import the relevant package
    ImageClassifier,
)

# Initialize the GUI with the layout and model list
models = ["model_b1", "model_b2", "model_b3", "model_b4", "model_b7"]
# Initialize paths for the model and class map
default_model_path = "models/model_b7"
class_map_path = "scripts/imagenet_class_map.txt"

# Instantiate the image classifier with the default model
model = ImageClassifier(default_model_path, class_map_path)


content = ""
img_path = "placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=15vw|>

<|{content}|file_selector|extensions=.jpeg|>
Select an image from your file system

<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""


def on_change(state, var_name, var_val):
    if var_name == "content":
        # Assuming var_val is the path to the image
        file_path = var_val
        try:
            # Open the image and process it for classification
            with open(file_path, "rb") as img_file:
                image_data = img_file.read()
            predicted_class, confidence = model.classify(image_data)

            state.prob = round(confidence * 100)
            state.pred = f"This is a {predicted_class}"

            # Update the image display with the path to the temporary file
            # Note: This depends on whether Taipy GUI can display images from local paths
            state.img_path = file_path
        except Exception as e:
            state.pred = f"An error occurred: {str(e)}"


app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)
