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
import pty
import sys

__version__ = "0.0.11-alpha"

_MOTD = r"""
                    .--,       .--,
                    ( (  \.---./  ) )
                     '.__/o   o\__.'
                       `{=  ^  =}´
                         >  u  <
 ____________________.""`-------`"".______________________  
\   ___                   __         __   _____       __  /
/  / _ \ ___  ___  ___   / /  ___ _ / /  / ___/__ __ / /_ \
\ / // // -_)/ -_)/ _ \ / /__/ _ `// _ \/ /__ / // // __/ /
//____/ \__/ \__// .__//____/\_,_//_.__/\___/ \_,_/ \__/  \
\_________________________________________________________/
                       ___)( )(___ `-.___. 
                      (((__) (__)))      ~`

Welcome to DeepLabCut docker!
"""


def _parse_args():
    parser = argparse.ArgumentParser(
        "deeplabcut-docker",
        description=(
            "Utility tool for launching DeepLabCut docker containers. "
            "Only a single argument is given to specify the container type. "
            "By default, the current directory is mounted into the container "
            "and used as the current working directory. You can additionally "
            "specify any additional docker argument specified in "
            "https://docs.docker.com/engine/reference/commandline/cli/."
        ),
    )
    parser.add_argument(
        "container",
        type=str,
        choices=["notebook", "bash"],
        help=(
            "The container to launch. A list of all containers is available on "
            "https://hub.docker.com/r/deeplabcut/deeplabcut/tags. By default, the "
            "latest DLC version will be selected and automatically updated, if "
            "possible. All containers are currently launched in interactive mode "
            "by default, meaning you can use Ctrl+C in your terminal session to "
            "terminate a command."
        ),
    )
    return parser.parse_known_args()


def main():
    """Main entry point. Parse arguments and launch container."""
    launch_args, docker_arguments = _parse_args()
    argv = ["deeplabcut_docker.sh", launch_args.container, *docker_arguments]
    print(_MOTD, file=sys.stderr)
    pty.spawn(argv)
    print("Container stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
