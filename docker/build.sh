#!/bin/bash
# Build script for deeplabcut docker images.
# > docker/build.sh [build|test|push]

set -e

DOCKER=${DOCKER:-'docker'}
DOCKER_BUILD="$DOCKER build"
BASENAME=deeplabcut/deeplabcut
DOCKERDIR=docker

if [[ ! -d ./${DOCKERDIR} ]]; then
    >&2 echo Run from repository root. Current pwd is
    >&2 pwd
    exit 1
fi

build_test_image() {
image_id=$1
${DOCKER_BUILD} -t deeplabcut:tmp - << EOF
from ${image_id}
run pip install --no-cache-dir pytest
run mkdir -p /app
run chmod a+rwx /app
EOF
}

list_images() {
    $DOCKER images \
    | grep '^deeplabcut ' \
    | sed -s 's/\s\s\+/\t/g' \
    | cut -f1,2 -d$'\t' --output-delimiter ':' \
    | grep core
}

run_test() {
    test_image_id="$1"
    test_image_id=$(build_test_image $test_image_id)

    kwargs=(
        -u $(id -u) --tmpfs /.local --tmpfs /.cache -w /app
        --env DLClight=True
        -it deeplabcut:tmp
    )

    # Unit tests
    $DOCKER run -v $(pwd)/tests:/app ${kwargs[@]} python3 -m pytest -q .

    # Functional tests
    $DOCKER run \
     -v $(pwd)/testscript_cli.py:/app/testscript_cli.py:ro \
     -v $(pwd)/examples:/app/examples:ro \
     ${kwargs[@]} \
     python3 testscript_cli.py
}

iterate_build_matrix() {
    ## TODO(stes): Consider adding legacy versions for CUDA
    ## if there is demand from users:
    # 10.2-runtime-ubuntu18.04 \
    # 11.1.1-runtime-ubuntu18.04 \
    for cuda_version in \
        11.4.0-runtime-ubuntu20.04 \
        11.7.0-runtime-ubuntu20.04
    do
        for deeplabcut_version in \
            2.2.0.2 \
            2.2.1.1
        do
            for stage in base core gui jupyter; do
                tag=${deeplabcut_version}-${stage}-cuda${cuda_version}
                echo \
                     --build-arg=CUDA_VERSION=${cuda_version} \
                     --build-arg=DEEPLABCUT_VERSION=${deeplabcut_version} \
                     "--tag=${BASENAME}:$tag" \
                     -f "Dockerfile.${stage}" \.
            done
        done
    done
}

for arg in "$@"; do
case $1 in
    build)
        cp -r examples ${DOCKERDIR}
        (
          cd ${DOCKERDIR}
          iterate_build_matrix \
          | tr '\n' '\0' \
          | xargs -I@ -0 bash -c "echo Building @; $DOCKER build -q @"
        )
    ;;
    test)
        for tag in latest-core; do
            run_test ${BASENAME}:${tag}
        done
    ;;
    push)
        for tag in base latest-core latest-gui latest-gui-jupyter; do
            ${DOCKER} push deeplabcut/deeplabcut:${tag}
        done
esac
done

