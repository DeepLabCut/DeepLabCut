#!/bin/bash
#
# Helper script for launching deeplabcut docker UI containers
# Usage:
#   $ ./deeplabcut-docker.sh [gui|notebook|bash]

DOCKER=${DOCKER:-docker}
DLC_VERSION=${DLC_VERSION:-"latest"}
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

# Select docker parameters based on the system.
# Display variable and bind paths slightly differ
# between macOS and Linux. Further systems should
# be added here.
get_x11_args() {
    if [[ $(uname -s) == Linux ]]; then
        err "Using Linux config"
        args=(
         "-e DISPLAY=unix$DISPLAY"
         "-v /tmp/.X11-unix:/tmp/.X11-unix" 
         "-v $XAUTHORITY:/home/developer/.Xauthority"
        )
    elif [[ $(uname -s) == Darwin ]]; then
        err "Using OSX config"
        # TODO(stes) This is most likely not robust for all users;
        #            We need to replace "en0" by some dynamic way
        #            of figuring out the active network interface.
        #            Even better would be to use 127.0.0.1, if this
        #            is possible with the correct external config
        ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
        display_id=$(echo $DISPLAY | sed -e 's/.*\(:[0-9]\)/\1/')
        args=(
         "-e DISPLAY=${ip}${display_id}"
        )
    else
        err "Unknown operating system:"
        err "$(uname -s)"
        err "Please open an issue on "
        err "https://github.com/DeepLabCut/DeepLabCut/issues"
        err "And attach your full console output."
        exit 1
    fi
    echo "${args[@]}"
}

get_mount_args() {
    args=(
        "-v $(pwd):/app -w /app" 
    )
    echo "${args[@]}"
}

get_container_name() {
    echo deeplabcut/deeplabcut:${DLC_VERSION}-$1
}

get_local_container_name() {
    echo deeplabcut-${DLC_VERSION}-$1
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
# Usage: build [core|gui|gui-jupyter]
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
    $DOCKER build -q -t ${local_name} - << EOF
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

# Launch the UI version of DeepLabCut
gui() {
    extra_args="$@"
    update gui || exit 1
    build gui || exit 1
    args="$(get_x11_args) $(get_mount_args) ${extra_args}"
    $DOCKER run -it --rm ${args} $(get_local_container_name gui) \
        || err "Failed to launch the DLC GUI. Used args: \"${args}\""
}

# Launch a Jupyter Server in the current directory
notebook() {
    extra_args="$@"
    update gui-jupyter || exit 1
    build gui-jupyter || exit 1
    args="$(get_x11_args) $(get_mount_args) ${extra_args} -v /app/examples"
    err "Starting the notebook server."
    err "Open your browser at"
    err "http://127.0.0.1:${DLC_NOTEBOOK_PORT}"
    err "If prompted for a password, enter 'deeplabcut'."
    $DOCKER run -p 127.0.0.1:${DLC_NOTEBOOK_PORT}:8888 -it --rm ${args} $(get_local_container_name gui-jupyter) \
        || err "Failed to launch the notebook server. Used args: \"${args}\""
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
subcommand=${1:-gui}
shift 1
case "${subcommand}" in
    gui) gui "$@" ;;
    notebook) notebook "$@" ;;
    bash) bash "$@" ;;
    custom) custom "$@" ;;
    *)
        echo "Usage"
        echo "$0 [gui|notebook|bash|help]"
        ;;
esac
