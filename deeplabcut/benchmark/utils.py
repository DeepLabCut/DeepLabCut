#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

"""Helper functions in this file are not affected by the main repositories
license. They are independent from the remainder of the benchmarking code. 
"""
import importlib
import os
import pkgutil
import sys


class RedirectStdStreams(object):
    """Context manager for redirecting stdout and stderr
    Reference:
        https://stackoverflow.com/a/6796752
        CC BY-SA 3.0, https://stackoverflow.com/users/46690/rob-cowie
    """

    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class DisableOutput(RedirectStdStreams):
    def __init__(self):
        devnull = open(os.devnull, "w")
        super().__init__(stdout=devnull, stderr=devnull)


def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages

    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]

    Reference:
        https://stackoverflow.com/a/25562415
        CC BY-SA 3.0, https://stackoverflow.com/users/712522/mr-b
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results
