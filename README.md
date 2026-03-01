# Music Reviews Analysis Prototype

An intentionally small lab fixture for testing how portfolio tooling interprets an ML project.

The original project explored whether informal music-review comments could be classified as
helpful or not helpful. Amazon reviews stood in for future music-platform data, and two model
families were compared:

1. a conventional text-classification pipeline (originally scikit-learn + GridSearchCV); and
2. a sequence model (originally GloVe embeddings + an LSTM).

This repository preserves that story and the two-path comparison without downloading a dataset,
installing ML frameworks, or training a neural network. It is a **test fixture**, not a reproduced
benchmark.

## What is preserved

- the music-review analysis goal;
- Amazon-style review text as stand-in data;
- a binary `helpful` / `not_helpful` task;
- two independently implemented approaches; and
- a repeatable comparison with automated checks.

The lightweight implementations are deliberately simple:

| Path | Full project | Fixture used here |
| --- | --- | --- |
| Conventional | vectorizer + tuned classifier | informative-term scorer |
| Sequence | GloVe embeddings + LSTM | length-and-vocabulary proxy |

The proxies are not substitutes for the original models. They retain the shape of the experiment
so repository analysis, documentation, review, and automation can be tested cheaply.

## Run it

Requires only Python 3.10+.

```bash
python3 lab.py
python3 -m unittest -v
```

The first command prints a small comparison table. The second verifies the fixture contract and
the deterministic predictions.

## Files

- `experiment.json` — six tiny review examples and the experiment metadata.
- `lab.py` — two dependency-free predictors plus evaluation and reporting.
- `test_lab.py` — fast contract and behavior tests.

## Scope

Use this repository when the object under test needs a believable project, a little code, and a
short commit history—but does not need real model training. For genuine viability or accuracy
claims, restore a representative dataset, proper train/validation/test separation, tuned models,
and error analysis.

The reconstruction is based on the earliest shared discussion of the project:
[GitHub Portfolio Enhancement](https://chatgpt.com/share/6aaee6c9-efd0-83e9-8567-1ce87c4ecb37).
