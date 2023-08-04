from abc import ABC, abstractmethod


class BaseProject(ABC):
    """
    Base class for project configuration.

    This class defines the basic structure and methods that a project configuration should have.
    Subclasses should implement the abstract method `convert2dict()` to convert the project configuration to a dictionary.
    """

    def __init__(self):
        pass

    @abstractmethod
    def convert2dict(self) -> dict:
        """Summary:
        Not yet implemented.
        Abstract method to convert the project configuration to a dictionary.

        Raises:
            NotImplementedError: This method must be implemented in the derived classes.
        """
        raise NotImplementedError

    @staticmethod
    def annotation2key(annotation):
        """Summary:
        Convert the annotation to a key.

        Args:
            annotation: The annotation to be converted.

        Returns:
            annotation: the project configuration as a dictionary.
        """
        return annotation


class BaseDataset(ABC):
    """
    Base class for datasets.

    This class defines the basic structure for datasets and serves as a superclass for future implementations.
    Subclasses should implement specific functionalities for their datasets.
    """

    pass
