# Music Review Helpfulness Classifier

A compact natural-language-processing prototype for identifying whether an informal review
contains enough concrete information to help another listener make a decision.

The project uses Amazon-style product reviews as a stand-in for the short, subjective comments
found on music platforms. Its purpose is to compare two different signals for helpfulness before
adapting the pipeline to music reviews at larger scale.

## Research question

Can a small text-classification pipeline distinguish detailed observations from low-information
reactions such as “love it” or “not for me”?

The current experiment treats this as a balanced binary classification task:

- `helpful` — includes concrete observations, conditions, or measurable details;
- `not_helpful` — expresses a reaction without enough supporting information.

## Approaches

### Conventional text signal

The conventional path scores review text using informative domain terms. It represents the
bag-of-words family of models and establishes an interpretable baseline.

### Sequence-aware text signal

The second path evaluates review length and vocabulary diversity. It captures structural cues
that are independent of individual keywords and provides a contrasting model for comparison.

These two paths establish the experiment structure for later evaluation with a tuned
scikit-learn classifier and a GloVe/LSTM sequence model.

## Prototype results

| Model path | Accuracy |
| --- | ---: |
| Conventional term evidence | 100% |
| Sequence-aware structure | 83% |

The comparison uses six balanced examples defined in `experiment.json`. Each run is deterministic,
which makes changes to preprocessing and decision rules easy to evaluate.

## Run the project

Python 3.10 or newer is required. There are no third-party dependencies.

```bash
python3 lab.py
```

Expected output:

```text
Music Review Helpfulness Classifier
Review examples: 6

path          accuracy
------------  --------
conventional     100%
sequence          83%
```

Run the regression suite with:

```bash
python3 -m unittest -v
```

## Project structure

```text
.
├── README.md          # project overview and results
├── DEVELOPMENT.md     # experiment decisions and milestones
├── experiment.json    # model metadata and labeled review examples
├── lab.py             # preprocessing, prediction, evaluation, and reporting
└── test_lab.py        # contract and behavior tests
```

## Development direction

The next stage is to replace the rule-based implementations with trained models, expand the
corpus with informal music comments, and compare precision, recall, F1, calibration, and error
types. Particular attention will be given to sarcasm, genre-specific language, very short expert
comments, and reviews that mix objective detail with strong subjective reactions.
