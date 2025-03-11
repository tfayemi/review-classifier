# data_processing.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_dataset(json_path):
    """
    Reads the dataset from a JSON file, where each line is a separate JSON object.

    Args:
        json_path (str): Path to the JSON file containing the review data.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded dataset.
    """
    df = pd.read_json(json_path, lines=True)
    print(f"Dataset loaded with shape: {df.shape}")

    # Print columns for reference
    for index, column in enumerate(df.columns):
        print(f"Column-{index} - {column}")

    return df


def find_helpful_ratio(helpful_list):
    """
    Calculates the helpfulness ratio for each review.
    Returns 1 if (helpful_count / total_count) >= 0.5, otherwise 0.
    If total_count is 0, returns 0.

    Args:
        helpful_list (list[int]): A two-element list [helpful_count, total_count].

    Returns:
        int: 1 if helpful ratio >= 0.5, else 0.
    """
    if helpful_list[1] > 0:
        return 1 if helpful_list[0] / helpful_list[1] >= 0.5 else 0
    else:
        return 0


def preprocess_traditional_ml(df, test_size=0.25, random_state=1, plot_distribution=True):
    """
    Prepares the DataFrame for the traditional ML pipeline by:
      1. Computing the helpfulness ratio (helpful vs total votes).
      2. Optionally plotting its distribution.
      3. Splitting the data into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame containing 'reviewText' and 'helpful' columns.
        test_size (float): Fraction of data to use for the test set.
        random_state (int): Seed for the random number generator (reproducibility).
        plot_distribution (bool): If True, plots the distribution of the helpfulness ratio.

    Returns:
        tuple: (X_train, X_test, Y_train, Y_test) for model training.
    """
    # Calculate the helpfulness ratio
    df['helpfulness_raatio'] = df['helpful'].apply(find_helpful_ratio)
    print("Sample of DataFrame after computing helpfulness ratio:")
    print(df.head())

    # Optionally plot the distribution of helpfulness
    if plot_distribution:
        df.hist(column='helpfulness_raatio', bins=50)
        plt.title("Distribution of Helpfulness Ratio")
        plt.show()

    # Split into features (reviews) and labels (helpfulness ratio)
    reviews = df['reviewText']
    labels = df['helpfulness_raatio']
    print("Total labels:", len(labels))

    # Perform train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        reviews, labels, test_size=test_size, random_state=random_state
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, X_test, Y_train, Y_test
