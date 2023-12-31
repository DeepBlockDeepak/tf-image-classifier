import ast
import json
import os
from typing import Dict

import tensorflow as tf


class ImageClassifier:
    def __init__(self, model_path: str, class_map_path: str):
        self.model = self.load_model(model_path)
        self.class_map = self.read_in_class_map(class_map_path)

    # load and return the model from the given path
    @staticmethod
    def load_model(model_path):
        if os.path.exists(model_path):
            # load the model using tf's load_model function
            return tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

    # load and preprocess and return the image
    @staticmethod
    def load_image(image_path, img_size=(224, 224)):
        if os.path.exists(image_path):
            # read the image file
            img = tf.io.read_file(image_path)
            # decode the image file to a tensor
            img = tf.image.decode_jpeg(img, channels=3)
            # resize the image to the size expected by the model
            img = tf.image.resize(img, img_size)
            # normalize pixel values to the range [0, 1]
            img = img / 255.0
            # add a batch dimension
            img = tf.expand_dims(img, axis=0)

            return img
        else:
            raise FileNotFoundError(f"Image path {image_path} does not exist.")

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
            # use ast module due to bad formatting in source file
            class_map = ast.literal_eval(data)

        return class_map

    # performs the final classification work
    def classify(self, image_path):
        img = self.load_image(image_path)
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
