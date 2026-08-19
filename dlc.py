"""
DeepLabCut2.0-2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import warnings

from deeplabcut import cli
from deeplabcut.core.deprecation import DLCDeprecationWarning


def main():
    warnings.warn(
        "Running `python dlc.py ...` is deprecated. Use the `dlc` command "
        "instead (e.g. `dlc train-network ...`), or `python -m deeplabcut.cli`.",
        DLCDeprecationWarning,
        stacklevel=2,
    )
    cli.main()


if __name__ == "__main__":
    main()
