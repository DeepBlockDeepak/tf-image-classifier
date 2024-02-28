import ast
import base64
import os
from typing import Dict, Union

import tensorflow as tf


class ImageClassifier:
    def __init__(self, model_path: str, class_map_path: str):
        self.model = self.load_model(model_path)
        self.class_map = self.read_in_class_map(class_map_path)

    # load and return the model
    @staticmethod
    def load_model(model_path):
        if tf.io.gfile.exists(model_path):
            # load the model using tf's load_model function
            return tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

    @staticmethod
    def load_image_from_memory(image_data, img_size=(224, 224)):
        # decode the image file to a tensor
        img = tf.io.decode_image(image_data, channels=3, expand_animations=False)
        # resize, normalize, and add a batch dimension as before
        img = tf.image.resize(img, img_size)
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)
        return img

    # predicts the class of the image
    def predict_image_class(self, img):
        # make predictions on the image
        predictions = self.model.predict(img)
        # index of the class with the highest predicted probability
        predicted_class = tf.argmax(predictions[0]).numpy()
        # highest predicted probability
        confidence = tf.reduce_max(predictions[0]).numpy()

        return predicted_class, confidence

    # Returns the top-5 predictions and their confidences with help from tf.nn.top_k
    def predict_top_image_classes(self, img, top_k=5):
        # make predictions on the image
        predictions = self.model.predict(img)[0]
        # indices of the classes with the highest predicted probabilities
        predicted_classes = tf.nn.top_k(predictions, k=top_k).indices.numpy()
        # highest predicted probabilities
        confidences = tf.nn.top_k(predictions, k=top_k).values.numpy()

        return predicted_classes, confidences

    # open the local txt file and load it into a dictionary
    # source: https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt
    @staticmethod
    def read_in_class_map(class_map_path) -> Dict[str, int]:
        with open(class_map_path, "r") as f:
            data = f.read()
            # use ast module due to formatting of source file
            class_map = ast.literal_eval(data)

        return class_map

    def classify(self, image_data: Union[str, bytes]):
        img = self.load_image_from_memory(image_data)
        predicted_class, confidence = self.predict_image_class(img)
        # get the associated animal "value" from the "key"-predicted_class
        predicted_class = self.class_map[predicted_class]
        return predicted_class, confidence

    # performs a prediction of the top-5 classes
    def classify_top_k(self, image_path, top_k=5):
        img = self.load_image(image_path)
        predicted_classes, confidences = self.predict_top_image_classes(img, top_k)
        # get the associated "values" from the "key"-predicted_classes
        predicted_classes = [self.class_map[i] for i in predicted_classes]

        return predicted_classes, confidences

    @staticmethod
    def print_prediction(model_str, image_path, predicted_class, confidence):
        model_name = "EfficientNet" + model_str.split("_")[1].upper()
        image_name = os.path.basename(image_path).split(".")[0]

        print("\n" * 2)
        print(f"Model Used: {model_name}")
        print(f"Image Classified: {image_name}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
        print("\n" * 2)

    @staticmethod
    def print_top_predictions(model_str, image_path, predicted_classes, confidences):
        model_name = "EfficientNet" + os.path.basename(model_str).split("_")[1].upper()
        image_name = os.path.basename(image_path).split(".")[0]

        print("\n" * 2)
        print(f"Model Used: {model_name}")
        print(f"Image Classified: {image_name}")
        for idx, pred_class in enumerate(predicted_classes):
            print(
                f"Prediction {idx}: {pred_class} (Confidence: {confidences[idx]:.2f})"
            )
        print("\n" * 2)
