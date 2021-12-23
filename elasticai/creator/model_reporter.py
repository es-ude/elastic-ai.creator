import csv
from typing import Sequence, Iterable, Any
import torch

from elasticai.creator.protocols import Model


class ModelReport:
    """
    The `ModelReport` aims to ease the verification of neural networks
    implemented in hardware. Given a model and a corresponding data
    provider it will produce a csv file containing the model's inference
    results on said data. Additionally it will store the paths to the
    inferred data samples so that another tool can use the data samples
    and the produced csv file to check that the hw model produces the
    same output as the software model for every data sample.
    """

    def __init__(
        self,
        model: Model,
        data: Iterable[Sequence[str, Any, Any]],
        is_binary: bool,
        threshold: float = 0.5,
    ) -> None:
        model.eval()
        self.model = model
        self.data: Iterable[Sequence[str, Any, Any]] = data
        self.evaluate(is_binary, threshold)

    def evaluate(self, is_binary: bool, threshold: float = 0.5) -> None:
        # TODO: add support for more labels
        """
        Create a results tuple from the input model and data, it is designed to be used in classification
        Args:
            is_binary (bool): Binary labels are handled slightly differently so it needs a bool value, currently only supported if True
            threshold (float): If binary the threshold to compare the model to
        """
        if is_binary:
            self._results = [
                (
                    example_id,
                    int(expected_value),
                    int(self.model(torch.unsqueeze(input_value, 1)) > threshold),
                )
                for example_id, input_value, expected_value in zip(
                    self.data[0], self.data[1], self.data[2]
                )
            ]

    def write_to_csv(self, path: str) -> None:
        """
        Writes the results to the path in a csv format
        Args:
            path(str): the results path eg "results.csv"
        """
        with open(path, "wt") as results_file:
            wtr = csv.writer(results_file, delimiter=" ", lineterminator="\n")
            wtr.writerow(["path", "ground-truth", "model-prediction"])
            wtr.writerows(self._results)

    def write_to_txt(self, path: str) -> None:
        """
        Writes the results to the path in a txt format
        Args:
            path(str): the results path eg "results.txt"
        """
        with open(path, "wt") as results_file:
            results_file.write("path ground-truth model-prediction\n")
            for result in self._results:
                results_file.write("{} {} {}\n".format(result[0], result[1], result[2]))
