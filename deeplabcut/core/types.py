class DeprecatedArgument:
    """Singleton sentinel class for deprecated arguments.

    Use this as a default value to distinguish between "argument not provided"
    and "argument explicitly set to None".

    Usage:
        from deeplabcut.core.types import DEPRECATED_ARGUMENT, DeprecatedArgument

        def func(old_arg=DEPRECATED_ARGUMENT):
            if isinstance(old_arg, DeprecatedArgument):
                # old_arg was not provided
            else:
                # old_arg was explicitly provided (even if None)
    """

    __slots__ = ()
    _instance: "DeprecatedArgument | None" = None

    def __new__(cls) -> "DeprecatedArgument":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<deprecated argument>"


DEPRECATED_ARGUMENT = DeprecatedArgument()
