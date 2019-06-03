"""
Author: Isaac Robinson

Module includes methods useful to loading all plugins placed in a folder, or module
"""

from typing import Set
from typing import Type
from typing import TypeVar
from types import ModuleType
import pkgutil
import sys

# Needed to actually load modules in folders containing plugins
def _load_modules(dirmodule: ModuleType, reload = True):
    """
    Loads all modules in a given package, or directory. Private method used by main plugin loader method.

    :param dirmodule: A module object, representing directory module(package) all packages are in...
    :param reload: Boolean, determines if modules should be reloaded if they already exist in sys.modules.
                   Defaults to True.
    """
    # Get absolute and relative package paths for this module...
    path = list(iter(dirmodule.__path__))[0]
    rel_path = dirmodule.__name__

    # Iterate all modules in specified directory using pkgutil, importing them if they are not in sys.modules
    for importer, package_name, ispkg in pkgutil.iter_modules([path]):
        # Name to look for in sys.modules...
        mod_name = rel_path + "." + package_name

        # If the module does not exist yet or reload flag is true, reload it.
        if((mod_name not in sys.modules) or (reload)):
            module = importer.find_module(package_name).load_module(package_name)
            sys.modules[mod_name] = module
        else:
            module = sys.modules[mod_name]

        # If the module is a subpackage, import all modules in that package(via recursive call)
        if(ispkg):
            _load_modules(module, reload)
    # Modules loaded, return...
    return


# Generic type for method below
T = TypeVar("T")

def load_plugin_classes(plugin_dir: ModuleType, plugin_metaclass: Type[T], reload: bool = True) -> Set[Type[T]]:
    """
    Loads all plugins, or classes, within the specified module folder that extend the provided metaclass type.

    :param plugin_dir: A module object representing the path containing plugins... Can get a module object
                       using import...
    :param plugin_metaclass: The metaclass that all plugins extend. Please note this is the class type, not the
                             instance of the class, so if the base class is Foo just type Foo as this argument.
    :param reload: Boolean, Determines if plugins should be reloaded if they already exist. Defaults to True.

    :return: A list of class types that directly extend the provided base class and where found in the specified
             module folder.
    """
    # Load the modules plugins could be in, so as to load the classes that might exist there into sys.modules.
    _load_modules(plugin_dir, reload)

    # Return all subclasses of the plugin base class as a set so as to remove duplicates.
    return set(plugin_metaclass.__subclasses__())


