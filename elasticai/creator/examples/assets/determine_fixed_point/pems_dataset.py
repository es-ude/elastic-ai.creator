from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

Slice = tuple[int | None, int | None]


@dataclass
class PeMSDataset(Dataset):
    """
    The dataset can be downloaded from this site (pems-4w.csv is used): https://zenodo.org/record/3939793
    """

    dataset_path: str | Path
    datapoints_per_sample: int
    sensor_idx: int
    raw_data_slice: Optional[Slice] = field(default=None)

    def __post_init__(self) -> None:
        assert (
            self.datapoints_per_sample > 0
        ), "datapoints_per_sample must be greater than 0."
        self.data = self._load_data()

    def __len__(self) -> int:
        n_samples = len(self.data) - (self.datapoints_per_sample + 1) + 1
        return n_samples if n_samples > 0 else 0

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.data[idx : idx + self.datapoints_per_sample]
        sample = sample.reshape((self.datapoints_per_sample, 1))
        label = self.data[idx + self.datapoints_per_sample]
        label = np.array([label], dtype=self.data.dtype)
        return sample, label

    def _load_data(self) -> np.ndarray:
        full_data = np.loadtxt(self.dataset_path, delimiter=",", dtype=np.float32)
        sensor_data = full_data[self.sensor_idx]
        if self.raw_data_slice is not None:
            start, end = self.raw_data_slice
            sensor_data = sensor_data[start:end]
        return sensor_data
