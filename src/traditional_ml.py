# traditional_ml.py

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def define_models():
    """
    Defines one or more model pipelines along with their hyperparameter grids.
    Returns a list of (pipeline, param_grid) tuples.
    """
    # Example pipeline using CountVectorizer -> TfidfTransformer -> SGDClassifier
    model_svm = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf-svm', SGDClassifier()),
    ])

    parameters_svm = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf-svm__alpha': (1e-2, 1e-3),
    }

    return [(model_svm, parameters_svm)]


def tune_model(model_tuple, X_train, y_train, metric='accuracy', cv=5):
    """
    Tunes a given model using GridSearchCV and returns the best model and score.

    Args:
        model_tuple (tuple): (pipeline, param_grid)
        X_train (array-like): Training features (reviews).
        y_train (array-like): Training labels (helpful or not helpful).
        metric (str): Scoring metric for GridSearchCV (default='accuracy').
        cv (int): Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_score)
    """
    pipeline, param_grid = model_tuple
    clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=metric)
    clf.fit(X_train, y_train)
    best_score = clf.best_score_
    best_model = clf
    print(f"Best Score: {best_score:.4f} with params: {clf.best_params_}")
    return (best_model, best_score)


def get_best_models_and_parameters(X_train, y_train, metric='accuracy', cv=5):
    """
    Iterates through all defined models, tunes them, and returns a sorted list
    of (model, score) tuples in descending order of score.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        metric (str): Scoring metric for GridSearchCV (default='accuracy').
        cv (int): Number of cross-validation folds.

    Returns:
        list of tuples: Each tuple is (best_model, best_score).
    """
    final_models = []
    models = define_models()
    for model_tuple in models:
        best_model_and_score = tune_model(model_tuple, X_train, y_train, metric=metric, cv=cv)
        final_models.append(best_model_and_score)

    # Sort the list of (model, score) in descending order of score
    final_list = sorted(final_models, key=lambda x: x[1], reverse=True)
    return final_list


def evaluate_traditional_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data, prints metrics,
    and plots a confusion matrix.

    Args:
        model: A fitted scikit-learn model (e.g., GridSearchCV or Pipeline).
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
    """
    preds = model.predict(X_test)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, preds)
    labels_list = ['helpful (1)', 'not helpful (0)']  # Adjust if reversed
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels_list, rotation=45)
    ax.set_yticklabels([''] + labels_list)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Print accuracy and classification report
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy:.4f}")
    cls_report = classification_report(y_test, preds)
    print("Classification Report:")
    print(cls_report)
