from typing import Any, Callable

import torch

from elasticai.creator.vhdl.evaluators.evaluator import Evaluator


class MetricEvaluator(Evaluator):
    def __init__(
        self,
        inference_evaluator: Evaluator,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | float],
    ) -> None:
        self.inference_evaluator = inference_evaluator
        self.metric = metric

    def run(self) -> Any:
        predictions, labels = self.inference_evaluator.run()
        return self.metric(predictions, labels)
