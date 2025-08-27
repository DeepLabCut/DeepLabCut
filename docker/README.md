# DeepLabCut Dockerfiles

**Note that this README is mainly intended for DeepLabCut developers. The main
documentation contains its own user documentation on the provided docker images.**

This repo contains build routines for the following official DeepLabCut docker images:
- `deeplabcut/deeplabcut:${DLC_VERSION}-base-cuda${CUDA_VERSION}-cudnn9`: Base image with DLC
- `deeplabcut/deeplabcut:${DLC_VERSION}-core-cuda${CUDA_VERSION}-cudnn9`: DLC in light mode
- `deeplabcut/deeplabcut:${DLC_VERSION}-jupyter-cuda${CUDA_VERSION}-cudnn9`: DLC with jupyter installed

All images come with Python 3.11 installed.
The images are synced to DockerHub: https://hub.docker.com/r/deeplabcut/deeplabcut

## Quickstart

### `deeplabcut-docker`

You can use the images fully standalone, without the need of cloning the DeepLabCut
repo. A helper package called `deeplabcut-docker` is available on PyPI and can be
installed by running:

```bash
pip install deeplabcut-docker
```

We provide docker containers for two different use cases outlined below. In both cases,
your current directory will be mounted in the container, and the container will be
started with your current username and group.

- Interactive console with DLC in light mode
  ```bash
  deeplabcut-docker bash
  ```
- A Jupyter notebook server can be launched with
  ```bash
  deeplabcut-docker notebook
  ```

You can pass `docker run` arguments to `deeplabcut-docker` directly. So if you have GPUs
and want them to be available in your Docker container, call:

```bash
deeplabcut-docker bash --gpus all
```

If you want to mount other volumes to your container, you can do so with the [`-v`
](https://docs.docker.com/reference/cli/docker/container/run/#volume) flag, as you would
when calling `docker run`:

```bash
deeplabcut-docker bash --gpus all -v /home/john:/home/john
```

You can select which DeepLabCut version and CUDA version to use through the 
`DLC_VERSION` and `CUDA_VERSION` environment variables. So to launch a container with 
CUDA 12.1 and DLC 3.0.0, you can run: 

```bash
DLC_VERSION=3.0.0 CUDA_VERSION=12.1 deeplabcut-docker bash --gpus all
```

*Note: Advanced users can also directly download and use the `deeplabcut-docker.sh`
script if this is preferred over a python helper script.*

### Jupyter Notebooks Running on Remote Servers

Sometimes, we want to run Jupyter Notebooks on remote servers but connect to them 
through the browser on our local machine. To do so, port forwarding needs to be used.
This is straightforward, and there are many resources you can explore on how to do so (
such as [this StackOverflow post](https://stackoverflow.com/a/69244262) or the [Jupyter 
Notebook docs](https://jupyter-notebook.readthedocs.io/en/4.x/public_server.html)).

This can easily be done with `deeplabcut-docker`. To run a DeepLabCut notebook on a
remote server:

```bash
# The Jupyter Server is running on port 8888 in the docker container
# You forward your server's port XXXX to the container's port 8888
# You forward port your laptop's port YYYY to port XXXX on the server
ssh -L localhost:YYYY:localhost:XXXX john@123.456.78.987
DLC_NOTEBOOK_PORT=XXXX deeplabcut-docker notebook --gpus all

# Example with XXXX=8889, YYYY=8890
# 1. Connect to your server, using port forwarding
ssh -L localhost:8890:localhost:8889 john@123.456.78.987

# 2. On the remote server, use deeplabcut-docker to launch the container
DLC_NOTEBOOK_PORT=8889 deeplabcut-docker notebook --gpus all

# 3. Connect to the server running on your machine at http://127.0.0.1:8890!
```

### Using Docker without `deeplabcut-docker`

Docker images can also be run without the `deeplabcut-docker` package, for more expert
users. This is not the recommended, as many of the nice features (such as starting 
the container with the current user instead of root) won't be there.

The `core` image can simply be run by pulling the image and using `docker run`:

```bash
docker pull deeplabcut/deeplabcut:3.0.0-core-cuda11.8-cudnn9
docker run -it --rm --gpus all deeplabcut/deeplabcut:3.0.0-core-cuda11.8-cudnn9
```

The `jupyter` image cannot be run in the same way. Notebook servers cannot be run as 
the root user (which can be dangerous) without passing the `--allow-root` option, so
running `docker run deeplabcut/deeplabcut:3.0.0-jupyter-cuda11.8-cudnn9` will lead to an  
error (`Running as root is not recommended. Use --allow-root to bypass`). What you can 
do (and we do in the `deeplabcut-docker` package) is to build a docker image with the
`jupyter` image as a base. We would recommend doing this for the `core` images as well. 
You can create the `Dockerfile`:

```dockerfile
FROM deeplabcut/deeplabcut:3.0.0-jupyter-cuda11.8-cudnn9
ARG UID
ARG GID
ARG UNAME
ARG GNAME

# Create same user as on the host system
RUN mkdir -p /home
RUN mkdir -p /app
RUN groupadd -g ${GID} ${GNAME} || groupmod -o -g ${GID} ${GNAME}
RUN useradd -d /home -s /bin/bash -u ${UID} -g ${GID} ${UNAME}
RUN chown -R ${UNAME}:${GNAME} /home
RUN chown -R ${UNAME}:${GNAME} /app
WORKDIR /app

# Switch to the local user from now on
USER ${UNAME}
```

And then build and run:

```bash
docker build \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg UNAME=$(id -un) \
  --build-arg GNAME=$(id -gn) \
  -t my-dlc-image \
  .
docker run -p 127.0.0.1:8889:8888 -it --rm --gpus all my-dlc-image
```

## For developers

Make sure your docker daemon is running and navigate to the repository root directory.
You can build the images by running

```
docker/build.sh build
```

Note that this assumes that you have rights to execute `docker build` and `docker run` commands which requires either `sudo` access or membership in the `docker` group on your local machine. If you are not in the `docker` group, run the script with the environment variable `DOCKER="sudo docker"` set to override the default docker command.

Images can be verified by running

```
docker/build.sh test
``` 

Built images can be pushed to DockerHub by running

```
docker/build.sh push
``` 

## Prerequisites (if you don't have Docker installed already)

**(1)** Install Docker. See https://docs.docker.com/install/ & for Ubuntu: https://docs.docker.com/install/linux/docker-ce/ubuntu/
Test docker: 

    $ sudo docker run hello-world
    
 The output should be: ``Hello from Docker! This message shows that your installation appears to be working correctly.``

*if you get the error ``docker: Error response from daemon: Unknown runtime specified nvidia.`` just simply restart docker: 
  
       $ sudo systemctl daemon-reload
       $ sudo systemctl restart docker

    
**(2)** Add your user to the docker group (https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user)
Quick guide  to create the docker group and add your user: 
Create the docker group.

    $ sudo groupadd docker
Add your user to the docker group.

    $ sudo usermod -aG docker $USER

(perhaps restart your computer (best) or (at min) open a new terminal to make sure that you are added from now on)

## Acknowledgements

Ascii art in the MOTD is adapted from https://ascii.co.uk/art/mice and https://patorjk.com/software/taag/#p=display&f=Small%20Slant&t=DeepLabCut.

```
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
```
