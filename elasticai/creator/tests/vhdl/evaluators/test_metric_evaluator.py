import unittest
from typing import Callable

import torch

from elasticai.creator.vhdl.evaluators.metric_evaluator import MetricEvaluator


class InferenceEvaluatorMock:
    def __init__(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        self.predictions = predictions
        self.labels = labels

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictions, self.labels


def get_metric_evaluator(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
) -> MetricEvaluator:
    return MetricEvaluator(
        inference_evaluator=InferenceEvaluatorMock(predictions, labels), metric=metric
    )


def difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a - b


def summed_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).sum().item()


class MetricEvaluatorTest(unittest.TestCase):
    def test_metric_evaluator_summed_differences_empty_predictions_labels(self) -> None:
        evaluator = get_metric_evaluator(
            predictions=torch.tensor([]),
            labels=torch.tensor([]),
            metric=summed_difference,
        )
        actual = evaluator.run()
        target = 0
        self.assertEqual(target, actual)

    def test_metric_evaluator_differences(self) -> None:
        evaluator = get_metric_evaluator(
            predictions=torch.tensor([1, 4, 3]),
            labels=torch.tensor([2, 2, 3]),
            metric=difference,
        )
        actual = evaluator.run().detach().numpy().tolist()
        target = [-1, 2, 0]
        self.assertEqual(target, actual)

    def test_metric_evaluator_summed_differences(self) -> None:
        evaluator = get_metric_evaluator(
            predictions=torch.tensor([1, 4, 3]),
            labels=torch.tensor([2, 2, 3]),
            metric=summed_difference,
        )
        actual = evaluator.run()
        target = 1
        self.assertEqual(target, actual)
