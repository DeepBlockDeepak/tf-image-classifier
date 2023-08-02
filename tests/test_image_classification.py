import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.image_classification.model import ImageClassifier


class TestImageClassifier(unittest.TestCase):
    def setUp(self):
        self.model_path = "models/model_b4"
        self.class_map_path = "scripts/imagenet_class_map.txt"
        self.image_path = "images/stella_boston_terrier.jpg"
        self.classifier = ImageClassifier(self.model_path, self.class_map_path)

    def test_load_model(self):
        # make sure the model can be loaded
        self.assertIsNotNone(self.classifier.model)

    def test_load_image(self):
        # test that an image can be loaded and preprocessed correctly
        img = self.classifier.load_image(self.image_path)
        self.assertIsNotNone(img)

    def test_predict_image_class(self):
        # test that class prediction can be made
        img = self.classifier.load_image(self.image_path)
        predicted_class, confidence = self.classifier.predict_image_class(img)
        self.assertIsNotNone(predicted_class)
        self.assertIsNotNone(confidence)

    def test_classify(self):
        # see if classify method works correctly
        predicted_class, confidence = self.classifier.classify(self.image_path)
        self.assertIsNotNone(predicted_class)
        self.assertIsNotNone(confidence)

    def test_classify_top_k(self):
        # test classify_top_k method works correctly
        predicted_classes, confidences = self.classifier.classify_top_k(self.image_path)
        self.assertIsNotNone(predicted_classes)
        self.assertIsNotNone(confidences)


if __name__ == "__main__":
    unittest.main()
