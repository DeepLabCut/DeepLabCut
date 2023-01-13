from abc import ABC, abstractmethod


class BaseProject(ABC):
    """
    TODO
    """

    def __init__(self):
        pass

    @abstractmethod
    def convert2dict(self):
        raise NotImplementedError

    @staticmethod
    def annotation2key(annotation):
        return annotation


class BaseDataset(ABC):
    """
    TODO
    """
    pass
