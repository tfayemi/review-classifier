# deep_learning.py

import numpy as np
import string
from nltk.corpus import stopwords
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Input,
    Dropout,
    LSTM,
    Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_glove_vectors(glove_file_name):
    """
    Reads the GloVe vectors from a file and constructs:
        - words_to_index: dict mapping words to integer indices.
        - index_to_words: dict mapping integer indices to words.
        - word_to_vector_map: dict mapping words to their GloVe vector (numpy array).

    Args:
        glove_file_name (str): Path to the GloVe .txt file.

    Returns:
        tuple:
            words_to_index (dict): Word -> index
            index_to_words (dict): Index -> word
            word_to_vector_map (dict): Word -> embedding vector
    """
    with open(glove_file_name, 'r', encoding='utf8') as f:
        words = set()
        word_to_vector_map = {}
        for line in f:
            line = line.strip().split()
            current_word = line[0]
            words.add(current_word)
            word_to_vector_map[current_word] = np.array(line[1:], dtype=np.float64)
        i = 1
        words_to_index = {}
        index_to_words = {}
        for word in sorted(words):
            words_to_index[word] = i
            index_to_words[i] = word
            i += 1
    return words_to_index, index_to_words, word_to_vector_map


def cosine_similarity(u, v):
    """
    Computes the cosine similarity between two vectors u and v.

    Args:
        u (np.ndarray): Vector u.
        v (np.ndarray): Vector v.

    Returns:
        float: Cosine similarity value between -1 and 1.
    """
    dot = np.dot(u, v)
    l2_norm_u = np.sqrt(np.sum(u ** 2))
    l2_norm_v = np.sqrt(np.sum(v ** 2))
    return dot / (l2_norm_u * l2_norm_v)


def convert_sentence_to_indices(X, words_to_index, max_len):
    """
    Converts a list/Series of sentences into a 2D array of indices,
    where each sentence is truncated/padded to max_len.

    Args:
        X (pd.Series or list[str]): Collection of text reviews.
        words_to_index (dict): Dictionary mapping words to their indices.
        max_len (int): Maximum length (number of words) per review.

    Returns:
        np.ndarray: Matrix of shape (len(X), max_len) with word indices.
    """
    # Ensure X is a list-like structure we can iterate over by index
    if hasattr(X, "reset_index"):
        X = X.reset_index(drop=True)

    m = len(X)
    indices = np.zeros((m, max_len))
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))

    for i in range(m):
        tokens = X[i].lower().split()
        # Remove punctuation
        stripped = [word.translate(table) for word in tokens]
        # Filter out stopwords and non-alpha tokens
        words = [word for word in stripped if word.isalpha() and word not in stop_words]

        j = 0
        for w in words:
            idx = words_to_index.get(w)
            if idx is not None:
                indices[i, j] = idx
                j += 1
            if j >= max_len:
                break

    return indices


def trained_embedding_layer(words_to_index, word_to_vector_map):
    """
    Creates a Keras Embedding layer using pre-trained GloVe vectors.
    The layer is set to non-trainable to preserve the pre-trained weights.

    Args:
        words_to_index (dict): Dictionary mapping words to integer indices.
        word_to_vector_map (dict): Dictionary mapping words to their GloVe vectors.

    Returns:
        keras.layers.Embedding: A configured Keras Embedding layer.
    """
    vocab_len = len(words_to_index) + 1
    # Example word to get embedding dimension
    emb_dim = next(iter(word_to_vector_map.values())).shape[0]

    # Initialize embedding matrix
    embedding_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in words_to_index.items():
        embedding_vector = word_to_vector_map.get(word)
        if embedding_vector is not None:
            embedding_matrix[index, :] = embedding_vector

    # Create the embedding layer
    embedding_layer = Embedding(
        input_dim=vocab_len,
        output_dim=emb_dim,
        trainable=False
    )
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer


def helpfulness_model(input_shape, words_to_index, word_to_vector_map):
    """
    Builds an LSTM-based model for predicting the helpfulness of a review.

    Args:
        input_shape (tuple): Shape of the input (e.g., (max_len,)).
        words_to_index (dict): Word -> index mapping for embedding.
        word_to_vector_map (dict): Word -> GloVe vector mapping.

    Returns:
        keras.Model: A compiled Keras model with LSTM layers for classification.
    """
    # Define the input
    sentence_indices = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer using the pre-trained GloVe vectors
    embedding_layer = trained_embedding_layer(words_to_index, word_to_vector_map)
    embeddings = embedding_layer(sentence_indices)

    # Add LSTM layers and dropout
    X = LSTM(32, return_sequences=True)(embeddings)
    X = Dropout(rate=0.5)(X)
    X = LSTM(32, return_sequences=False)(X)
    X = Dropout(rate=0.5)(X)

    # Final dense layer for binary classification
    X = Dense(1, activation='sigmoid')(X)
    X = Activation('sigmoid')(X)

    # Build the model
    model = Model(inputs=sentence_indices, outputs=X)
    return model
