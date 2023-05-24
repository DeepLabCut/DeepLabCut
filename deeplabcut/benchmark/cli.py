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

"""Command line interface for DeepLabCut deeplabcut.benchmark."""

import argparse

import deeplabcut.benchmark


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="+", default=None, required=False)
    parser.add_argument(
        "--onerror",
        default="return",
        required=False,
        choices=("ignore", "return", "raise"),
    )
    parser.add_argument("--nocache", action="store_true")
    return parser.parse_args()


def main():
    """Main CLI entry point for generating deeplabcut.benchmark results."""
    args = _parse_args()
    if not args.nocache:
        results = deeplabcut.benchmark.loadcache()
    else:
        results = None
    results = deeplabcut.benchmark.evaluate(
        include_benchmarks=args.include,
        results=results,
        on_error=args.onerror,
    )
    if not args.nocache:
        deeplabcut.benchmark.savecache(results)
    try:
        print(results.toframe())
    except StopIteration:
        pass
