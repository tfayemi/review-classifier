# utils.py

"""
Utilities Module
----------------
This file contains general-purpose helper functions that can be reused
across different parts of the project (e.g., similarity measures,
text-cleaning helpers, logging wrappers, etc.).
"""

import numpy as np


def cosine_similarity(u, v):
    """
    Computes the cosine similarity between two vectors u and v.

    Args:
        u (np.ndarray): Vector u.
        v (np.ndarray): Vector v.

    Returns:
        float: Cosine similarity value in the range [-1, 1].
    """
    dot = np.dot(u, v)
    l2_norm_u = np.sqrt(np.sum(u ** 2))
    l2_norm_v = np.sqrt(np.sum(v ** 2))
    return dot / (l2_norm_u * l2_norm_v)

# If you have other small, common functions or classes that multiple modules
# might use, you can add them here. For example:
#
# def clean_text(text):
#     # Some text-cleaning steps used by both ML and DL pipelines
#     return cleaned_text
#
# def load_config(config_path):
#     # Load JSON or YAML configuration
#     return config_dict
