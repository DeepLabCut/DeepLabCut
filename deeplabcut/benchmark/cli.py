"""Command line interface for DeepLabCut benchmark."""

import argparse

import deeplabcut.benchmark


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="+", default=None, required=False)
    parser.add_argument(
        "--onerror", 
        default="return", 
        required=False, 
        choices=("ignore", "return", "raise")
    )
    parser.add_argument("--nocache", action="store_true")
    return parser.parse_args()


def main():
    """Main CLI entry point for generating benchmark results."""
    args = _parse_args()
    if not args.nocache:
        results = benchmark.loadcache()
    else:
        results = None
    results = benchmark.evaluate(
        include_benchmarks=args.include,
        results=results,
        on_error=args.onerror,
    )
    if not args.nocache:
        benchmark.savecache(results)
    print(results.toframe())
