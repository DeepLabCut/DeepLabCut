FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update -yy \ 
    && apt-get install -yy --no-install-recommends python3 python3-pip ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --upgrade "deeplabcut>=2.2.0.2" numpy==1.19.5 decorator==4.4.2 tensorflow==2.5.0

# TODO required to fix permission errors when running the container with limited permission.
RUN chmod a+rwx -R /usr/local/lib/python3.8/dist-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained
