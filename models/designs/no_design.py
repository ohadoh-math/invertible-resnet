"""
A stub design.
"""


from .design import Design, Dataset, Module


class NoDesign(Design):
    """
    Don't perform any design.
    """

    def __init__(self, dataset: Dataset):
        self._ds = dataset

    def get_dataset(self) -> Dataset:
        return self._ds

    def update_design(self, model: Module):
        pass
