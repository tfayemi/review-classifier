"""
Review Classifier
-----------------
This package contains modules for loading and preprocessing data,
running traditional ML pipelines, and building deep learning models
to classify review helpfulness.
"""

__version__ = "0.1.0"

# Optionally, import selected functions/classes from your submodules
# so they can be accessed as `from review_classifier.src import load_dataset` etc.
# If you prefer explicit imports, you can skip these or import only what you need.

from .data_processing import (
    load_dataset,
    find_helpful_ratio,
    preprocess_traditional_ml
)

from .traditional_ml import (
    define_models,
    tune_model,
    get_best_models_and_parameters,
    evaluate_traditional_model
)

from .deep_learning import (
    read_glove_vectors,
    cosine_similarity,
    convert_sentence_to_indices,
    trained_embedding_layer,
    helpfulness_model
)

# If you want to specify exactly what gets imported when someone does:
# `from review_classifier.src import *`
# you can define an __all__ list:

__all__ = [
    "load_dataset",
    "find_helpful_ratio",
    "preprocess_traditional_ml",
    "define_models",
    "tune_model",
    "get_best_models_and_parameters",
    "evaluate_traditional_model",
    "read_glove_vectors",
    "cosine_similarity",
    "convert_sentence_to_indices",
    "trained_embedding_layer",
    "helpfulness_model",
]
