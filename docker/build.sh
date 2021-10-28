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
    $DOCKER images | grep '^deeplabcut ' | sed -s 's/\s\s\+/\t/g' | cut -f1,2 -d$'\t' --output-delimiter ':' | grep core
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

for arg in "$@"; do
case $1 in
    build)
	cp -r examples ${DOCKERDIR}
        for tag in base latest-core latest-gui latest-gui-jupyter; do
            (
            cd ${DOCKERDIR};
            ${DOCKER_BUILD} -t ${BASENAME}:${tag} -f Dockerfile.${tag} .
            )
            echo $tag
        done
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

