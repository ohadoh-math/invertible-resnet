"""
A core-set design as described by:
    https://arxiv.org/abs/1708.00489
"""


import logging
import numpy
import torch
import torchvision
from torchvision import transforms
from  torch.utils.data import DataLoader, Subset
from  torch.utils.data.dataloader import default_collate
from .design import Design, Module, Dataset
from .k_centers_greedy_acquisition import KCenterGreedyAcquisitionLowRank


def flattening_collate(data_entries):
    return default_collate([(x[0].flatten(), x[1]) for x in data_entries])


class CoresetDesignKCentersGreedy(Design):
    """
    A design that extracts a core-set from the dataset.
    The algorithm used is algorithm 1 ("k-Centers Greedy") in the cited paper.
    """

    def __init__(self, dataset: Dataset, design_size: int):
        self._dataset = dataset
        self._design_size = design_size
        self._indices = numpy.arange(len(self._dataset))
        self._loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            num_workers=2,
            collate_fn=flattening_collate,
        )
        self._index_count = 0
        self._design_ds = None
        self._indices = []

        self._do_design()

    @property
    def indices(self):
        return self._indices[:self._index_count]

    @property
    def remaining_indices(self):
        return self._indices[self._index_count:]

    def get_dataset(self) -> Dataset:
        assert self._design_ds is not None
        return self._design_ds

    def update_design(self, model: Module):
        # currently we won't update the design
        pass

    def _pick_uniform_indices(self, k):
        """
        Pick `k` indices uniformly from the remaining indices.
        """
        remaining_indices = self.remaining_indices
        assert k <= len(remaining_indices)

        numpy.random.shuffle(remaining_indices)
        self._index_count += k

        return self._indices[self._index_count-k:self._index_count]

    def _do_design(self):
        """
        Pick a covering set according to the algorithm 1 ("k-Centers Greedy").
        """
        # load the data into one giant matrix
        n = len(self._dataset)
        d = len(self._dataset[0][0].flatten())  # the first 0 is the dataset item, which is a tuple of image and label. the second extracts the image
        X = torch.zeros(n, d, device='cpu')

        cursor = 0
        for batch, _labels in self._loader:
            X[cursor:cursor+len(batch)] = batch
            cursor += len(batch)

        logging.info("X[0] = %r", X[0])
        logging.info("X[last] = %r", X[len(X)-1])

        k_center_finder = KCenterGreedyAcquisitionLowRank(X)
        self._indices = [
            k_center_finder.next()
            for _ in range(self._design_size)
        ]

        logging.info("design set = %r", self._indices)
        self._design_ds = Subset(self._dataset, self._indices)


def _main():
    from models.utils_cifar import train, test, std, mean, get_hms, interpolate

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

    train_chain = [
        transforms.Pad(4, padding_mode="symmetric"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    clf_chain = [transforms.Normalize(mean['cifar10'], std['cifar10'])]
    transform_train = transforms.Compose(train_chain + clf_chain)
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    logging.info("loaded cifar10 (%d images)", len(trainset))

    design = CoresetDesignKCentersGreedy(trainset, 100)

    logging.info("design size = %d", len(design.dataset))


if __name__ == "__main__":
    _main()
