"""
Author: Isaac Robinson

Module includes methods useful to loading all plugins placed in a folder, or module
"""

from typing import Set
from typing import Type
from typing import TypeVar
from abc import ABC
import sys
import pkgutil

# Needed to actually load modules in folders containing plugins
def _load_modules(dirname: str):
    """
    Loads all modules in a given package, or directory. Private method used by main plugin loader method

    :param dirname: Path to the directory, can be relative.
    """
    # Iterate all modules in specified directory using pkgutil, importing them if they are not in sys.modules
    for importer, package_name, ispkg in pkgutil.iter_modules([dirname]):
        full_pkg_name = f"{dirname}.{package_name}"

        if(full_pkg_name not in sys.modules):
            importer.find_module(package_name).load_module(package_name)

    return


# Generic type for method below
T = TypeVar("T")

def load_plugin_classes(plugin_dir: str, plugin_metaclass: Type[T]) -> Set[Type[T]]:
    """
    Loads all plugins, or classes, within the specified module folder that extend the provided metaclass type.

    :param plugin_dir: The directory or package all modules or plugins are in, written in dot notation
                       Ex: "foo.bar.baz" for directory "foo/bar/baz" in the project...
    :param plugin_metaclass: The metaclass that all plugins extend. Please note this is the class type, not the
                             instance of the class, so if the base class is Foo just type Foo as this argument.

    :return: A list of class types that directly extend the provided base class and where found in the specified
             module folder.
    """
    # Load the modules plugins could be in, so as to load the classes that might exist there into sys.modules.
    _load_modules(plugin_dir)

    # Return all subclasses of the plugin base class as a set so as to remove duplicates.
    return set(plugin_metaclass.__subclasses__())


