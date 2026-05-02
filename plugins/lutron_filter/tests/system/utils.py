import csv
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


def create_testbench_data(model: Module, test_set: Dataset, run_dir: Path) -> None:
    gen = _TestBenchDataCreator(model, test_set, run_dir)
    gen.run()


class _TestBenchDataCreator:
    def __init__(self, model: Module, test_set: Dataset, run_dir: Path) -> None:
        self._model = model
        self._test_set = test_set
        self._run_dir = run_dir
        self._binarize_layers: list[str] = []

    def run(self) -> None:
        self._setup()
        results = self._predict()
        for name, data in results.items():
            self._store_data_to_csv(f"test_data_{name}.csv", data)
        self._store_result_after_each_precomp_block()

    @property
    def tb_path(self) -> Path:
        return self._run_dir / "testbench"

    def _store_data_to_csv(self, file_name: str, data: Iterable):
        with open(self.tb_path / file_name, "w") as f:
            writer = csv.writer(f)
            for item in data:
                writer.writerow(item)

    def _setup(self) -> None:
        self._model.to("cpu")
        model = self._model
        binarize_layers = []
        for name, layer in model.named_modules():
            if name.startswith("binarize"):
                binarize_layers.append(name)
                _make_layer_record_output(layer)
        self._binarize_layers = binarize_layers
        model.eval()

    def _predict(self) -> dict[str, Iterable]:
        predictions = []
        test_set = self._test_set
        model = self._model
        testloader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

        def unify_pair_first_element(x):
            if isinstance(x, tuple) or (isinstance(x, list) and len(x) in (1, 2)):
                return x[0]
            return x

        samples = [
            sample.permute((0, 2, 1)).flatten().to(int).numpy()
            for sample in map(unify_pair_first_element, testloader)
        ]
        with torch.no_grad():
            for sample in map(unify_pair_first_element, testloader):
                sample = sample.to("cpu")
                prediction = model(sample)
                predicted_label = torch.where(prediction < 0.5, 0, 1)
                predictions.append(predicted_label[0].numpy())
        return {"samples": samples, "predictions": predictions}

    def _store_result_after_each_precomp_block(self):
        binarize_layers = self._binarize_layers
        num_samples = len(self._test_set)  # type: ignore

        for name in binarize_layers:
            layer = getattr(self._model, name)
            results = torch.stack(layer.results)
            results = (results + 1) / 2
            results = results.to(torch.int)
            # flattened = results.swapdims(1, 2)
            flattened = results.permute((0, 1, 3, 2))
            flattened = flattened.reshape(num_samples, -1)

            flattened = flattened.tolist()
            tmp = []
            tmp.extend("".join(str(int(x)) for x in xs) for xs in flattened)
            with open(self.tb_path / "out_{}.txt".format(name), "w") as f:
                for i, r in enumerate(tmp):
                    f.write(r)
                    f.write(" : {}".format(results[i].tolist()))
                    f.write
                    f.write("\n")


def _make_layer_record_output(layer):
    layer.results = []
    old_forward = layer.forward

    def forward(x: torch.Tensor):
        y: torch.Tensor = old_forward(x)
        result = y.detach()
        layer.results.append(result)
        return y

    setattr(layer, "forward", forward)
