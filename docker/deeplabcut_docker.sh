#!/bin/bash
#
# Helper script for launching deeplabcut docker UI containers
# Usage:
#   $ ./deeplabcut-docker.sh [notebook|bash]

DOCKER=${DOCKER:-docker}
CUDA_VERSION=${CUDA_VERSION:-"12.4"}
CUDNN_VERSION=${CUDNN_VERSION:-"9"}
DLC_VERSION=${DLC_VERSION:-"3.0.0"}
DLC_NOTEBOOK_PORT=${DLC_NOTEBOOK_PORT:-8888}

# Check if the current users has privileges to start
# a docker container.
check_system() {
    if [[ $(uname -s) == Linux ]]; then
        if [ $(groups | grep -c docker) -eq 0 ]; then
            if [[ "$DOCKER" == "sudo docker" ]]; then
                return 0
            fi
            err "The current user $(id -u) is not                      "
            err "part of the \"docker\" group.                         "
            err "Please either:                                        "
            err " 1) Launch this script with the DOCKER environment    "
            err "    variable set to DOCKER=\"sudo docker\" (use this  "
            err "    with care)!                                       "
            err " 2) Add your user to the docker group. You might need "
            err "    to log in and out again to see the effect of the  "
            err "    change.                                           "
            exit 1
        fi
    elif [[ $(uname -s) == Darwin ]]; then
        err "Please note that macOSX support is currently experimental"
        err "If you encounter errors, please open an issue on"
        err "https://github.com/DeepLabCut/DeepLabCut/issues"
        err "Thanks for testing the package!"
    fi
}

get_mount_args() {
    args=(
        "-v $(pwd):/app -w /app"
    )
    echo "${args[@]}"
}

get_container_name() {
    echo "deeplabcut/deeplabcut:${DLC_VERSION}-$1-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}"
}

get_local_container_name() {
    echo "deeplabcut-${DLC_VERSION}-$1-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}"
}

### Start of helper functions ###

# Print error messages to stderr
# Ref. https://google.github.io/styleguide/shellguide.html#stdout-vs-stderr
err() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

# Update the docker container
update() {
    $DOCKER pull $(get_container_name $1)
}

# Build the docker container
# Usage: build [core|jupyter]
build() {
    tag=$1
    _build $(get_container_name $tag) $(get_local_container_name $tag) || exit 1
}

_build() {
    remote_name=$1
    local_name=$2

    uname=$(id -un)
    uid=$(id -u)
    gname=$(id -gn)
    gid=$(id -g)

    err "Configuring a local container for user $uname ($uid) in group $gname ($gid)"
    $DOCKER build -q -t "${local_name}" - <<EOF
    from ${remote_name}

    # Create same user as on the host system
    run mkdir -p /home
    run mkdir -p /app
    run groupadd -g $gid $gname || groupmod -o -g $gid $gname
    run useradd -d /home -s /bin/bash -u $uid -g $gid $uname
    run chown -R $uname:$gname /home
    run chown -R $uname:$gname /app

    # Switch to the local user from now on
    user $uname
EOF
    if [ $? -ne 0 ]; then
        err Build failed.
        exit 1
    fi
    err Build succeeded
}

### Start of CLI functions ###

# Launch a Jupyter Server in the current directory
notebook() {
    extra_args="$@"
    update jupyter || exit 1
    build jupyter || exit 1
    args="$(get_mount_args) ${extra_args}"
    err "Starting the notebook server."
    err "Open your browser at"
    err "http://127.0.0.1:${DLC_NOTEBOOK_PORT}"
    err "If prompted for a password, enter 'deeplabcut'."
    $DOCKER run -p 127.0.0.1:"${DLC_NOTEBOOK_PORT}":8888 -it --rm ${args} $(get_local_container_name jupyter) ||
        err "Failed to launch the notebook server. Used args: \"${args}\""
}

# Launch the command line, using DLC in light mode
bash() {
    extra_args="$@"
    update core || exit 1
    build core || exit 1
    args="$(get_mount_args) ${extra_args}"
    $DOCKER run -it $args $(get_local_container_name core) bash
}

# Launch a custom docker image (for developers)
# Takes a local image name as the first argument.
custom() {
    image=$1
    shift 1
    extra_args="$@"
    _build $image "${image}-custom" || exit 1
    args="$(get_mount_args) ${extra_args}"
    $DOCKER run -it $args ${image}-custom bash
}

check_system
subcommand=${1:-notebook}
shift 1
case "${subcommand}" in
notebook) notebook "$@" ;;
bash) bash "$@" ;;
custom) custom "$@" ;;
*)
    echo "Usage"
    echo "$0 [notebook|bash|help]"
    ;;
esac
