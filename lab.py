"""Run a tiny, deterministic version of the original two-model experiment."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

LABEL_HELPFUL = "helpful"
LABEL_NOT_HELPFUL = "not_helpful"
EXPERIMENT_PATH = Path(__file__).with_name("experiment.json")

INFORMATIVE_TERMS = {
    "battery",
    "clamping",
    "hours",
    "fit",
    "separation",
    "soundstage",
    "vocals",
}


def tokens(text: str) -> list[str]:
    """Return lowercase word tokens without external NLP packages."""
    return re.findall(r"[a-z0-9]+", text.lower())


def conventional_proxy(text: str) -> str:
    """Approximate a bag-of-words path using informative-term evidence."""
    evidence = INFORMATIVE_TERMS.intersection(tokens(text))
    return LABEL_HELPFUL if len(evidence) >= 2 else LABEL_NOT_HELPFUL


def sequence_proxy(text: str) -> str:
    """Approximate a sequence path using detail and vocabulary diversity."""
    words = tokens(text)
    diversity = len(set(words)) / len(words) if words else 0.0
    return LABEL_HELPFUL if len(words) >= 8 and diversity >= 0.70 else LABEL_NOT_HELPFUL


def load_experiment(path: Path = EXPERIMENT_PATH) -> dict:
    """Load the small declarative experiment fixture."""
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(examples: list[dict[str, str]], predict: Callable[[str], str]) -> float:
    """Calculate accuracy for a predictor over the fixture examples."""
    correct = sum(predict(example["text"]) == example["label"] for example in examples)
    return correct / len(examples)


def compare(experiment: dict) -> list[tuple[str, float]]:
    """Evaluate both lightweight paths in the same order as the manifest."""
    predictors = {
        "conventional": conventional_proxy,
        "sequence": sequence_proxy,
    }
    examples = experiment["examples"]
    return [
        (model["id"], accuracy(examples, predictors[model["id"]]))
        for model in experiment["models"]
    ]


def main() -> None:
    experiment = load_experiment()
    print(experiment["name"])
    print(f"Fixture examples: {len(experiment['examples'])}\n")
    print("path          accuracy")
    print("------------  --------")
    for model_id, score in compare(experiment):
        print(f"{model_id:<12}  {score:>7.0%}")


if __name__ == "__main__":
    main()
