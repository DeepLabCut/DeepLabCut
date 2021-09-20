#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut2.0-2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import argparse
import subprocess
import sys

__version__ = '0.0.1-alpha'

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'container',
        type = str,
        choices = ["gui", "notebook", "bash"],
    )
    return parser.parse_args()

def main():
    # https://stackoverflow.com/a/18422264
    # CC BY-SA 4.0, Viktor Kerkez & Smart Manoj
    args = _parse_args()
    process = subprocess.Popen(['./deeplabcut-docker.sh', args.container], 
                stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()
    #for c in iter(lambda: process.stdout.read(1), b''): 
    #    sys.stdout.buffer.write(c)

if __name__ == '__main__':
    main()