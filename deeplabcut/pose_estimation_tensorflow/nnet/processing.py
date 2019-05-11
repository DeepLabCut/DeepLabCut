"""
Author: Isaac Robinson

Contains Abstract Base Class for all predictor plugins, which when provided probability frames from the neural net,
figure out where the points should be in the image. They are executed when deeplabcut.analyze_videos is run...
"""
# Abstract class stuff
from abc import ABC
from abc import abstractmethod

# Used for type hints
from numpy import ndarray
from typing import List
from typing import Union


class Predictor(ABC):
    """
    Base plugin class for all predictor plugins.

    Predictors accept a source map of data received
    """
    # TODO: Add abstract processing methods and contructor all plugins must override
    # TODO: Add documentation...
    @abstractmethod
    def __init__(self, bodyparts: List[str]):

        pass

    @abstractmethod
    def on_frames(self, scmap: ndarray) -> Union[None, ndarray]:
        pass

    @abstractmethod
    def on_end(self) -> Union[None, ndarray]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this predictor plugin, name is used when selecting a predictor in deeplabcut.analyze_videos

        :return: The name of this plugin to be used to select it, as a string.
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get the description of this plugin, the equivalent of a doc-string for this plugin, is displayed when
        user lists available plugins

        :return: The description/summary of this plugin.
        """
        pass