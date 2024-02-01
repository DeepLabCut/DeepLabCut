#!/bin/bash
# Build script for deeplabcut docker images.
# > docker/build.sh [build|test|push]

set -e

# Set default Docker binary
export DOCKER=${DOCKER:-'docker'}
export DOCKER_BUILD="$DOCKER build"
export BASENAME=deeplabcut/deeplabcut
export DOCKERDIR=docker

# Check if script is being run from the correct directory
if [[ ! -d ./${DOCKERDIR} ]]; then
    echo >&2 Run from repository root. Current pwd is
    pwd >&2
    exit 1
fi

# List Docker images related to DeepLabCut
list_images() {
    $DOCKER images |
        grep '^deeplabcut ' |
        sed -s 's/\s\s\+/\t/g' |
        cut -f1,2 -d$'\t' --output-delimiter ':' |
        grep core
}

# Run tests inside Docker containers
run_test() {
    kwargs=(
        -u $(id -u) --tmpfs /.local --tmpfs /.cache
        --tmpfs /test/.pytest_cache
        --env DLClight=True -t
        $1
    )

    # Unit tests
    $DOCKER run ${kwargs[@]} python3 -m pytest -v tests || return 255

    # Functional tests
    $DOCKER run ${kwargs[@]} python3 testscript_cli.py || return 255

    return 0
}
export -f run_test

# Iterate through build matrix and perform actions
iterate_build_matrix() {
    ## TODO(stes): Consider adding legacy versions for CUDA
    ## if there is demand from users:

    #[add other dlc versions to build here]
    dlc_versions=(
        "2.3.5"
        #"2.3.2"
    )

    #[add other cuda versions to build here]
    cuda_versions=(
        #"11.4.3-cudnn8-runtime-ubuntu20.04"
        "11.7.1-cudnn8-runtime-ubuntu20.04"
    )

    docker_types=(
        "base"
        "test"
        "core"
        #"gui"
    )

    mode=${1:-build}
    for cuda_version in \
        ${cuda_versions[@]}; do
        for deeplabcut_version in \
            ${dlc_versions[@]}; do
            for stage in \
                ${docker_types[@]}; do
                tag=${deeplabcut_version}-${stage}-cuda${cuda_version}-latest
                case "$mode" in
                build)
                    echo \
                        --build-arg=CUDA_VERSION=${cuda_version} \
                        --build-arg=DEEPLABCUT_VERSION=${deeplabcut_version} \
                        "--tag=${BASENAME}:$tag" \
                        -f "Dockerfile.${stage}" \.
                    # --no-cache \
                    ;;
                push | clean | test)
                    echo ${BASENAME}:${tag}
                    ;;
                esac
            done
        done
    done
}

# Get Git hash
githash() {
    git log -1 --pretty=format:"%h"
}

# Create logs directory and set log file name
mkdir -p logs
logfile=logs/$(date +%y%m%d-%H%M%S)-$(githash)
echo "Logging to $logdir.*"

if [ $# -eq 0 ]; then
    echo "Help: Provide arguments to this script."
    echo "Usage: $0 [build|test|push]"
    exit 1
fi

# Iterate through command line arguments
for arg in "$@"; do
    case $1 in
    clean)
        iterate_build_matrix clean |
            tr '\n' '\0' |
            xargs -I@ -0 bash -c "docker image rm @ |& grep -v 'No such image'"
        ;;
    build)
        echo "DeepLabCut docker build:: $(git log -1 --oneline)"
        cp -r examples ${DOCKERDIR}
        (
            cd ${DOCKERDIR}
            iterate_build_matrix |
                tr '\n' '\0' |
                xargs -I@ -0 bash -c "echo Building @; $DOCKER build @ || exit 255"
            echo Successful build.
        ) |& tee ${logfile}.build
        ;;
    test)
        (
            echo "DeepLabCut docker build:: $(git log -1 --oneline)"
            iterate_build_matrix test |
                grep '\-test\-' |
                tr '\n' '\0' |
                xargs -0 -I@ bash -c "run_test @ || exit 255"
            echo Successful test.
        ) |& tee ${logfile}.test
        ;;
    push)
        iterate_build_matrix push |
            grep -v '\-test\-' |
            tr '\n' '\0' |
            xargs -I@ -0 bash -c "echo docker push @; \
				docker push @; \
				docker image rm @ |& grep -v 'No such image'"
        ;;
    *)
        echo "Usage: $0 [build|test|push]"
        exit 1
        ;;
    esac
done
