from setuptools import find_packages, setup

setup(
    name="tf-image-classifier",
    version="0.1",
    description="<...>",
    author="Jordan Medina",
    author_email="jordan.medina1729@gmail.com",
    url="https://github.com/DeepBlockDeepak/",
    install_requires=["tensorflow>=2.11"],
    classifiers=["Programming Language :: Python :: 3"],
    python_requires=">=3.7",
    packages=find_packages("./src"),
    package_dir={"": "src"},
    extras_require={
        "dev": ["pre-commit", "pytest", "sphinx"],
        "research": ["jupyterlab", "tensorflow_datasets", "wandb"],
    },
)
