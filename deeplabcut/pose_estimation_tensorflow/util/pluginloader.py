"""
Module includes methods useful to loading all plugins placed in a folder, or module...
"""

from typing import Set
from typing import Type
from typing import TypeVar
from types import ModuleType
import pkgutil
import sys


# Generic type for method below
T = TypeVar("T")

def load_plugin_classes(plugin_dir: ModuleType, plugin_metaclass: Type[T], do_reload: bool = True) -> Set[Type[T]]:
    """
    Loads all plugins, or classes, within the specified module folder and submodules that extend the provided metaclass
    type.

    :param plugin_dir: A module object representing the path containing plugins... Can get a module object
                       using import...
    :param plugin_metaclass: The metaclass that all plugins extend. Please note this is the class type, not the
                             instance of the class, so if the base class is Foo just type Foo as this argument.
    :param do_reload: Boolean, Determines if plugins should be reloaded if they already exist. Defaults to True.

    :return: A list of class types that directly extend the provided base class and where found in the specified
             module folder.
    """
    # Get absolute and relative package paths for this module...
    path = list(iter(plugin_dir.__path__))[0]
    rel_path = plugin_dir.__name__

    plugins: Set[Type[T]] = set()

    # Iterate all modules in specified directory using pkgutil, importing them if they are not in sys.modules
    for importer, package_name, ispkg in pkgutil.iter_modules([path]):
        mod_name = rel_path + "." + package_name

        # If the module name is not in system modules or the reload flag is set to true, perform a full load of the
        # modules...
        if((mod_name not in sys.modules) or do_reload):
            try:
                sub_module = importer.find_module(package_name).load_module(package_name)
                sys.modules[mod_name] = sub_module
            except Exception as e:
                print(f"Can't load '{mod_name}'. Due to issue below:")
                print(e)
                continue
        else:
            sub_module = sys.modules[mod_name]

        # Now we check if the module is a package, and if so, recursively call this method
        if(ispkg):
            plugins = plugins | load_plugin_classes(sub_module, plugin_metaclass, do_reload)
        else:
            # Otherwise we begin looking for plugin classes
            for item in dir(sub_module):
                field = getattr(sub_module, item)

                # Checking if the field is a class, and if the field is a direct child of the plugin class
                if(isinstance(field, type) and issubclass(field, plugin_metaclass) and (field != plugin_metaclass)):
                    # It is a plugin, add it to the list...
                    plugins.add(field)

    return plugins


