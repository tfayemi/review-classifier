"""Fast checks for the portfolio lab fixture."""

import unittest

import lab


class LabFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.experiment = lab.load_experiment()

    def test_experiment_keeps_two_distinct_model_paths(self) -> None:
        model_ids = [model["id"] for model in self.experiment["models"]]
        self.assertEqual(model_ids, ["conventional", "sequence"])

    def test_fixture_is_deliberately_small_and_balanced(self) -> None:
        examples = self.experiment["examples"]
        labels = [example["label"] for example in examples]
        self.assertEqual(len(examples), 6)
        self.assertEqual(labels.count(lab.LABEL_HELPFUL), 3)
        self.assertEqual(labels.count(lab.LABEL_NOT_HELPFUL), 3)

    def test_comparison_is_deterministic(self) -> None:
        self.assertEqual(
            lab.compare(self.experiment),
            [("conventional", 1.0), ("sequence", 5 / 6)],
        )


if __name__ == "__main__":
    unittest.main()
