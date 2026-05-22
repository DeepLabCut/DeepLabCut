#!/usr/bin/env python3
"""Helper CLI to run DeepLabCut Docker images (LGPL-3.0)."""

import argparse
import grp
import os
import platform
import pwd
import shlex
import subprocess
import sys
from datetime import datetime, timezone

__version__ = "0.0.12-alpha"

_IMAGE = "deeplabcut/deeplabcut"
_DEFAULT_CUDA = "12.4"


def _docker() -> list[str]:
    """Return the docker CLI argv prefix (from DOCKER env or `docker`)."""
    return shlex.split(os.environ.get("DOCKER", "docker"))


def _log(msg: str) -> None:
    """Log a timestamped message to stderr."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{ts}]: {msg}", file=sys.stderr)


def _check_system() -> None:
    """Verify docker group membership on Linux; warn on macOS."""
    if platform.system() == "Linux":
        if os.environ.get("DOCKER", "docker").strip() == "sudo docker":
            return
        if os.geteuid() == 0:
            return
        try:
            docker_gid = grp.getgrnam("docker").gr_gid
        except KeyError:
            return
        if docker_gid not in os.getgroups():
            _log(f'The current user {os.getuid()} is not in the "docker" group.')
            _log('Use DOCKER="sudo docker" (with care) or add your user to "docker".')
            sys.exit(1)
    elif platform.system() == "Darwin":
        _log("macOS support is experimental; report issues at")
        _log("https://github.com/DeepLabCut/DeepLabCut/issues")


def _remote_tag(mode: str) -> str:
    """Get the DockerHub image tag from DLC_VERSION and CUDA_VERSION env vars."""
    cuda = os.environ.get("CUDA_VERSION", _DEFAULT_CUDA)
    ver = os.environ.get("DLC_VERSION", "").strip()
    if mode == "notebook":
        if ver:
            return f"{_IMAGE}:{ver}-jupyter-cuda{cuda}"
        return f"{_IMAGE}:latest-jupyter"
    if ver:
        return f"{_IMAGE}:{ver}-core-cuda{cuda}"
    return f"{_IMAGE}:latest"


def _warn_if_not_jupyter_image(ref: str) -> None:
    """Warn if the image does not appear to have a Jupyter entrypoint."""
    r = subprocess.run(
        _docker()
        + [
            "image",
            "inspect",
            ref,
            "--format",
            "{{json .Config.Entrypoint}} {{json .Config.Cmd}}",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.exit(f"Could not inspect image {ref!r} after pull.\n{r.stderr.strip()}")
    blob = (r.stdout or "").lower()
    if "jupyter" not in blob:
        _log(
            f"Warning: image {ref!r} does not appear to have a Jupyter entrypoint. "
            "Proceeding anyway — if the server fails to start, ensure the image "
            "exposes a Jupyter-compatible entrypoint on port 8888."
        )


def _build_user_image(remote: str, local: str) -> None:
    """Build a small local image on top of remote with the current UID/GID user."""
    try:
        uid, gid = os.getuid(), os.getgid()
    except AttributeError:
        sys.exit("deeplabcut-docker requires a POSIX system (Linux or macOS).")
    user = pwd.getpwuid(uid).pw_name
    group = grp.getgrgid(gid).gr_name
    _log(f"Configuring a local image for user {user} ({uid}) in group {group} ({gid})")
    dockerfile = (
        "\n".join(
            (
                f"FROM {remote}",
                f"RUN mkdir -p /home/{user} /app",
                f"RUN groupadd -g {gid} {group} || groupmod -o -g {gid} {group}",
                f"RUN useradd -d /home/{user} -s /bin/bash -u {uid} -g {gid} {user}",
                f"RUN chown -R {user}:{group} /home/{user} /app",
                f"USER {user}",
            )
        )
        + "\n"
    )
    subprocess.run(
        _docker() + ["build", "-q", "-t", local, "-"],
        input=dockerfile.encode(),
        check=True,
    )
    _log("Build succeeded")


def _supplementary_group_args() -> list[str]:
    """Return --group-add flags for each supplementary group of the current user."""
    primary_gid = os.getgid()
    args = []
    for gid in os.getgroups():
        if gid != primary_gid:
            args += ["--group-add", str(gid)]
    return args


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args and return (namespace, extra args for docker run)."""
    parser = argparse.ArgumentParser(
        prog="deeplabcut-docker",
        description=(
            "Launch DeepLabCut Docker containers. The current directory is mounted "
            "at /app and used as the working directory. Additional arguments are "
            "passed through to `docker run` (see "
            "https://docs.docker.com/engine/reference/commandline/cli/)."
        ),
    )
    parser.add_argument(
        "container",
        choices=("notebook", "bash"),
        help=(
            "notebook: Jupyter server; bash: interactive shell. "
            "Image tags: https://hub.docker.com/r/deeplabcut/deeplabcut/tags — "
            "use DLC_VERSION and CUDA_VERSION to select a versioned tag; unset "
            "DLC_VERSION uses latest / latest-jupyter."
        ),
    )
    parser.add_argument(
        "--image",
        metavar="REF",
        help=(
            "Use this image (name:tag or digest) instead of the default from "
            "DLC_VERSION / CUDA_VERSION. For notebook, the image is checked for "
            "a Jupyter Notebook entrypoint after pull."
        ),
    )
    return parser.parse_known_args()


def _pull_image_if_not_exists(remote: str) -> None:
    """Pull the image if it does not exist locally."""
    _log(f"Pulling image {remote!r} if it does not exist locally...")
    r = subprocess.run(_docker() + ["image", "inspect", remote], capture_output=True)
    if r.returncode != 0:
        try:
            subprocess.run(_docker() + ["pull", remote], check=True)
        except subprocess.CalledProcessError:
            _log(
                f"Failed to pull image {remote!r}. Please verify that you specified "
                "a valid image name and tag, e.g. deeplabcut/deeplabcut:latest."
            )
            sys.exit(1)
    else:
        _log(f"Using local image {remote!r} (skipping pull)")


def main() -> None:
    """Entry point: pull, user-layer build, and run the container."""
    _check_system()
    args, docker_run_args = _parse_args()
    mode = args.container

    remote = args.image or _remote_tag(mode)
    local = f"deeplabcut-local-{mode}"
    _pull_image_if_not_exists(remote)
    if mode == "notebook":
        _warn_if_not_jupyter_image(remote)
    _build_user_image(remote, local)

    run = _docker() + ["run", "-it", "--rm", "-v", f"{os.getcwd()}:/app", "-w", "/app"]
    run += _supplementary_group_args()
    if mode == "notebook":
        port = os.environ.get("DLC_NOTEBOOK_PORT", "8888")
        token = os.environ.get("NOTEBOOK_TOKEN", "deeplabcut")
        _log("Starting the notebook server.")
        _log(f"Open your browser at http://127.0.0.1:{port}")
        if token == "deeplabcut":
            _log(f"If prompted for a token, enter {token!r}.")
        else:
            _log("If prompted for a token, enter the value of NOTEBOOK_TOKEN.")
        run += ["-p", f"127.0.0.1:{port}:8888", "-e", f"NOTEBOOK_TOKEN={token}"]
    run += docker_run_args + [local] + ([] if mode == "notebook" else ["bash"])
    sys.exit(subprocess.run(run).returncode)


if __name__ == "__main__":
    main()
