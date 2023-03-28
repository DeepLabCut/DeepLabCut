#!/bin/bash
# Build script for deeplabcut docker images.
# > docker/build.sh [build|test|push]

set -e

export DOCKER=${DOCKER:-'docker'}
export DOCKER_BUILD="$DOCKER build"
export BASENAME=deeplabcut/deeplabcut
export DOCKERDIR=docker

if [[ ! -d ./${DOCKERDIR} ]]; then
    >&2 echo Run from repository root. Current pwd is
    >&2 pwd
    exit 1
fi

list_images() {
    $DOCKER images \
    | grep '^deeplabcut ' \
    | sed -s 's/\s\s\+/\t/g' \
    | cut -f1,2 -d$'\t' --output-delimiter ':' \
    | grep core
}

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

iterate_build_matrix() {
    ## TODO(stes): Consider adding legacy versions for CUDA
    ## if there is demand from users:
    # 10.2-runtime-ubuntu18.04 \
    # 11.1.1-runtime-ubuntu18.04 \
    mode=${1:-build}
    for cuda_version in \
        11.4.0-cudnn8-runtime-ubuntu20.04 \
        11.7.0-cudnn8-runtime-ubuntu20.04
    do
        for deeplabcut_version in \
            2.2.0.6 \
            2.2.1.1
        do
            for stage in base core test gui jupyter; do
                tag=${deeplabcut_version}-${stage}-cuda${cuda_version}
                case "$mode" in
                    build)
                        echo \
                         --build-arg=CUDA_VERSION=${cuda_version} \
                         --build-arg=DEEPLABCUT_VERSION=${deeplabcut_version} \
                         "--tag=${BASENAME}:$tag" \
                         -f "Dockerfile.${stage}" \.
                    ;;
                    clean|test|push)
                        echo ${BASENAME}:${tag}
                    ;;
                esac
            done
        done
    done
}

githash() {
    git log -1 --pretty=format:"%h"
}

mkdir -p logs
logfile=logs/$(date +%y%m%d-%H%M%S)-$(githash)
echo "Logging to $logdir.*"

for arg in "$@"; do
case $1 in
    clean)
          iterate_build_matrix clean \
          | tr '\n' '\0' \
          | xargs -I@ -0 bash -c "docker image rm @ |& grep -v 'No such image'"
    ;;
    build)
        echo "DeepLabCut docker build:: $(git log -1 --oneline)"
        cp -r examples ${DOCKERDIR}
        (
          cd ${DOCKERDIR}
          iterate_build_matrix \
          | tr '\n' '\0' \
          | xargs -I@ -0 bash -c "echo Building @; $DOCKER build @ || exit 255"
          echo Successful build.
        ) |& tee ${logfile}.build
    ;;
    test)
        (
          echo "DeepLabCut docker build:: $(git log -1 --oneline)"
          iterate_build_matrix test \
          | grep '\-test\-' \
          | tr '\n' '\0' \
          | xargs -0 -I@ bash -c "run_test @ || exit 255"
          echo Successful test.
        ) |& tee ${logfile}.test
    ;;
    push)
        iterate_build_matrix push \
        | grep -v '\-test\-' \
        | tr '\n' '\0' \
        | xargs -I@ -0 echo ${DOCKER} push @
    ;;
esac
done

