# TF Image Classifier

TF Image Classifier is a showcase of some image classification code using TensorFlow, an open-source machine learning library. This project classifies [type-of-images, e.g., "dog breeds"] and serves as an exercise in deploying ML systems to production environments.

## Project Goals:
- Demonstrate proficiency in deploying machine learning systems to real-world production settings.
- Emphasize the packaging of ML software for ease of use and maintainability.


## Setup and Prerequisites

Before you can run the TF Image Classifier, there are a few local resources you need to set up:

1. **Models Directory**: 
    - This project utilizes models from TensorFlow Hub's EfficientNet. 
    - As an example, the model at [this link](https://tfhub.dev/tensorflow/efficientnet/b4/classification/1) was used and saved in this project as `/models/model_b4`.
    - My current structure of the `model_b4` directory:
        ```
        models/model_b4/
        ├── saved_model.pb
        ├── assets
        └── variables
            ├── variables.data-00000-of-00001
            └── variables.index
        ```

2. **Images Directory**:
    - This project requires test images stored in a directory named `images`.
    - Note: Please ensure that you have the right images in this directory before running the classifier.

3. **Wandb (Weights & Biases) Directory**:
    - This project uses Weights & Biases for experiment tracking. When you run an experiment script that initializes wandb (like the provided `baseline.py`), a `wandb` directory will be auto-generated in the directory from which you ran the script.
    - This directory caches data, checkpoints, and configuration related to your Weights & Biases runs. It's essential for ensuring smooth communication between your local machine and the Weights & Biases cloud.
    - You do not need to manually create this directory; it will be created for you upon the initialization of wandb in any script.


## Quick Start

### Example Run:
From the project root, run the classifier:
`python scripts/image_classifier_write_to_json.py models/model_b7 images/stella_boston_terrier.jpg`


### Testing

To run the entire test suite, use the command:
`python -m unittest discover tests`

For a single test:
`python -m unittest tests/test_image_classification.py`


## Want to Contribute?

Thank you for considering contributing to TF Image Classifier! If you'd like to contribute, please read our [CONTRIBUTING.md](CONTRIBUTING.md) guide to get started.

## License

[MIT](LICENSE.md)

