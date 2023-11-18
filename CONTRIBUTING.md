# Contributing to tf-classifier

## We Develop with Github

We use Github to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull Requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Setting up Development Environment

Here are the steps to setup the development environment:

1. Clone the repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd tf-image-classifier`
3. Install the package in editable mode with dev extras: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`
5. To update the hooks to the latest versions, run: `pre-commit autoupdate`


## Testing

To run the entire test suite, use the command `python -m unittest discover tests`. If you want to run a single test, you can do so by specifying the test file, like so: `python -m unittest tests/test_image_classification.py`.

## Use a Consistent Coding Style

We are using `black` for Python which enforces a consistent coding style. Run `black .` before committing to ensure your code follows the style guide.

## Documentation

Currently, the documentation for this project is contained within the code via docstrings and in this markdown file. Please ensure that you update relevant documentation when contributing code. In the future, we plan to support Sphinx for more comprehensive documentation.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/master/CONTRIBUTING.md)
