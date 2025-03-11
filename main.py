# main.py

from src.data_processing import load_dataset, preprocess_traditional_ml
from src.deep_learning import (
    read_glove_vectors,
    convert_sentence_to_indices,
    helpfulness_model
)
from src.traditional_ml import (
    get_best_models_and_parameters,
    evaluate_traditional_model
)
# Assuming cosine_similarity only lives in utils.py:
from src.utils import cosine_similarity


def main():
    # 1. Load and preprocess data (Traditional ML portion)
    df = load_dataset("./data/sample_dataset.json")
    X_train, X_test, Y_train, Y_test = preprocess_traditional_ml(df)

    # 2. Traditional ML Model
    best_models = get_best_models_and_parameters(X_train, Y_train)  # <-- Match variable names
    best_model = best_models[0][0]  # The best GridSearchCV object
    evaluate_traditional_model(best_model, X_test, Y_test)

    # 3. Deep Learning: Read GloVe vectors and build model
    words_to_index, index_to_words, word_to_vector_map = read_glove_vectors('./glove/glove.6B.100d.txt')
    model = helpfulness_model((50,), words_to_index, word_to_vector_map)

    # (Optional) Convert sentences to indices, compile, and train the DL model
    # X_train_indices = convert_sentence_to_indices(X_train, words_to_index, max_len=50)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train_indices, Y_train, epochs=5, batch_size=32, validation_split=0.2)
    # ... Evaluate model on test data, etc.

    # 4. Demonstrate or remove cosine_similarity usage
    # If you want to demonstrate, define vector1, vector2
    # For example, using GloVe vectors for "father" and "mother":
    # vector1 = word_to_vector_map["father"]
    # vector2 = word_to_vector_map["mother"]
    # similarity_score = cosine_similarity(vector1, vector2)
    # print("Cosine similarity:", similarity_score)


if __name__ == "__main__":
    main()
