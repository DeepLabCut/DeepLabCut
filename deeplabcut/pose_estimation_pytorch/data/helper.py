#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

from abc import ABCMeta


def cfg_getter(key, default=None):
    def _getter(cfg):
        return cfg.get(key, default)

    return _getter


def class_property(func, arg_func):
    """
    Decorator to create a class property.

    Parameters:
    - func: Callable that represents the logic of the property.
    - arg_func: Callable that provides the arguments for `func`.

    Returns:
    - A property with the logic encapsulated in `func` and arguments derived from `arg_func`.
    """

    def decorator_wrapper(method):
        def wrapper(self):
            return func(arg_func(self))

        return property(wrapper)

    return decorator_wrapper


class PropertyMeta(type):
    """
    Metaclass for creating class properties in a more organized and systematic manner.

    This metaclass allows a class to define its properties using a simple dictionary
    structure (`properties`). The dictionary keys represent the property names,
    while the values are tuples containing two callables:
    1. The function that represents the logic of the property.
    2. The function that provides the arguments for the logic function.

    Usage:
    class MyClass(metaclass=PropertyMeta):
        properties = {
            'property_name': (logic_function, arguments_function),
            # ... more properties ...
        }

    For each property specified in the `properties` dictionary, the metaclass will
    generate a real property that uses the logic from `logic_function` and
    arguments from `arguments_function`.

    Attributes:
    - properties (dict): Dictionary containing property names as keys and tuples
      of (logic_function, arguments_function) as values.
    """

    def __new__(cls, name, bases, attrs):
        if "properties" not in attrs:
            raise AttributeError(f"{name} must define a 'properties' dictionary.")
        properties = attrs.get("properties", {})
        for prop_name, (func, arg_func) in properties.items():
            attrs[prop_name] = class_property(func, arg_func)(lambda self: None)
        return super().__new__(cls, name, bases, attrs)


class CombinedPropertyMeta(ABCMeta, PropertyMeta):
    """
    Combined metaclass that integrates the functionalities of both `ABCMeta` and `BasePropertyMeta`.

    This metaclass is useful in scenarios where a class needs to use both abstract methods (from `ABCMeta`)
    and the property definition utilities provided by `BasePropertyMeta`.

    By using this metaclass, a class can be both an abstract class (with abstract methods and/or properties)
    and can also define properties in the structured manner facilitated by `PropertyMeta`.

    Inherits:
    - ABCMeta: Metaclass for base classes that include abstract methods.
    - PropertyMeta: Metaclass that facilitates structured property definitions.

    Note:
    When defining a class using `CombinedPropertyMeta`, ensure that the class also inherits
    from `ABC` to make it compatible with the `ABCMeta` behavior.
    """
