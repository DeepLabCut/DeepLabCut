#!/bin/bash

for tag in base latest-core latest-gui; do
    docker push deeplabcut/deeplabcut:${tag}
done
