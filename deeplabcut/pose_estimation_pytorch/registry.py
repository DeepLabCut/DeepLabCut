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
import inspect
from functools import partial
from typing import Any, Dict, Optional


def build_from_cfg(
    cfg: Dict, registry: "Registry", default_args: Optional[Dict] = None
) -> Any:
    """Builds a module from the configuration dictionary when it represents a class configuration,
    or call a function from the configuration dictionary when it represents a function configuration.

    Args:
        cfg: Configuration dictionary. It should at least contain the key "type".
        registry: The registry to search the type from.
        default_args: Default initialization arguments.
                      Defaults to None.

    Returns:
        Any: The constructed object.

    Example:
        >>> from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg
        >>> class Model:
        >>>     def __init__(self, param):
        >>>         self.param = param
        >>> cfg = {"type": "Model", "param": 10}
        >>> registry = Registry("models")
        >>> registry.register_module(Model)
        >>> obj = build_from_cfg(cfg, registry)
        >>> assert isinstance(obj, Model)
        >>> assert obj.param == 10
    """

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """A registry to map strings to classes or functions.
    Registered objects could be built from the registry. Meanwhile, registered
    functions could be called from the registry.

    Args:
        name: Registry name.
        build_func: Builds function to construct an instance from
                    the Registry. If neither ``parent`` nor
                    ``build_func`` is specified, the ``build_from_cfg``
                    function is used. If ``parent`` is specified and
                    ``build_func`` is not given,  ``build_func`` will be
                    inherited from ``parent``. Default: None.
        parent: Parent registry. The class registered in
                children's registry could be built from the parent.
                Default: None.
        scope: The scope of the registry. It is the key to search
               for children's registry. If not specified, scope will be the
               name of the package where the class is defined, e.g. mmdet, mmcls, mmseg.
               Default: None.

    Attributes:
        name: Registry name.
        module_dict: The dictionary containing registered modules.
        children: The dictionary containing children registries.
        scope: The scope of the registry.
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = "."

        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.
        The first scope will be split from key.
        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'
        Return:
            tuple[str | None, str]: The former element is the first scope of
            the key, which can be ``None``. The latter is the remaining key.
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record.

        Args:
            key: The class name in string format.

        Returns:
            class: The corresponding class.

        Example:
            >>> from deeplabcut.pose_estimation_pytorch.registry import Registry
            >>> registry = Registry("models")
            >>> class Model:
            >>>     pass
            >>> registry.register_module(Model, "Model")
            >>> assert registry.get("Model") == Model
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        """Builds an instance from the registry.

        Args:
            *args: Arguments passed to the build function.
            **kwargs: Keyword arguments passed to the build function.

        Returns:
            Any: The constructed object.

        Example:
            >>> from deeplabcut.pose_estimation_pytorch.registry import Registry, build_from_cfg
            >>> class Model:
            >>>     def __init__(self, param):
            >>>         self.param = param
            >>> cfg = {"type": "Model", "param": 10}
            >>> registry = Registry("models")
            >>> registry.register_module(Model)
            >>> obj = registry.build(cfg, param=20)
            >>> assert isinstance(obj, Model)
            >>> assert obj.param == 20
        """
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry.

        Args:
            registry: The registry to be added as children based on its scope.

        Returns:
            None

        Example:
            >>> from deeplabcut.pose_estimation_pytorch.registry import Registry
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> class Model:
            >>>     pass
            >>> mmdet_models.register_module(Model)
            >>> obj = models.build(dict(type='mmdet.Model'))
            >>> assert isinstance(obj, Model)
        """
        assert isinstance(registry, Registry)
        assert registry.scope is not None
        assert (
            registry.scope not in self.children
        ), f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    def _register_module(self, module, module_name=None, force=False):
        """Register a module.

        Args:
            module: Module class or function to be registered.
            module_name: The module name(s) to be registered.
                                                     If not specified, the class name will be used.
            force: Whether to override an existing class with the same name.
                                    Default: False.

        Returns:
            None

        Example:
            >>> from deeplabcut.pose_estimation_pytorch.registry import Registry
            >>> registry = Registry("models")
            >>> class Model:
            >>>     pass
            >>> registry._register_module(Model, "Model")
            >>> assert registry.get("Model") == Model
        """
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError(
                "module must be a class or a function, " f"but got {type(module)}"
            )

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered " f"in {self.name}")
            self._module_dict[name] = module

    def deprecated_register_module(self, cls=None, force=False):
        """Decorator to register a class in the registry.

        Args:
            cls: The class to be registered.
            force: Whether to override an existing class with the same name.
                                    Default: False.

        Returns:
            type: The input class.

        Example:
            >>> from deeplabcut.pose_estimation_pytorch.registry import Registry
            >>> registry = Registry("models")
            >>> @registry.deprecated_register_module()
            >>> class Model:
            >>>     pass
            >>> assert registry.get("Model") == Model
        """
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        Args:
            name: The module name to be registered. If not
                  specified, the class name will be used.
            force: Whether to override an existing class with
                   the same name. Default: False.
            module: Module class or function to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return
