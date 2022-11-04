from abc import ABC, abstractmethod

class Base(ABC):

    @abstractmethod
    def create_from_config(self, config):
        raise NotImplementedError


