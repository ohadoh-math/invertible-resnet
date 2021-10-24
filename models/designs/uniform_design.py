"""
A design that picks an arbitrary number of samples uniformly from the dataset.
Used as the baseline design.
"""


import numpy
from torch.utils.data import Subset
from .design import Design, Dataset, Module


class UniformDesign(Design):
    def __init__(self, dataset: Dataset, batch_size: int, update: bool = True):
        self._dataset = dataset
        self._batch_size = batch_size
        self._update = update
        self._remaining_indices = numpy.arange(len(dataset))
        self._indices = []
        self._design_ds = None

        self.update_design(None)

    def get_dataset(self) -> Dataset:
        return self._design_ds

    def update_design(self, _model: Module):
        if not self._update and self._indices:
            return self._design_ds

        target_size = min(len(self._dataset), len(self._indices) + self._batch_size)

        # shuffle the remaining indices and take the trailing `subset_size` indices
        numpy.random.shuffle(self._remaining_indices)
        picked_indices = self._remaining_indices[-self._batch_size:]
        self._remaining_indices = self._remaining_indices[:-self._batch_size]
        self._indices += list(picked_indices)
        assert len(self._indices) == target_size

        self._design_ds = Subset(self._dataset, self._indices)
