# DeepLabCut Dockerfiles

This repo contains build routines for the following official DeepLabCut docker images:
- `deeplabcut/deeplabcut:base`: Base image with TF2.5, cuDNN8 and DLC
- `deeplabcut/deeplabcut:latest-core`: DLC in light mode
- `deeplabcut/deeplabcut:latest-gui`: DLC in UI mode
- `deeplabcut/deeplabcut:latest-gui-jupyter`: DLC in UI mode, with jupyter installed

All images are based on Python 3.8.
The images are synced to DockerHub: https://hub.docker.com/r/deeplabcut/deeplabcut

## Quickstart

You can use the images fully standalone, without the need of cloning the DeepLabCut repo.
Simply download the `deeplabcut-docker.sh` script to your local working directory.
We provide docker containers for three different use cases outlined below.

In all cases, your current directory will be mounted in the container, and the container
will be started with your current username and group.

### User interface

To launch the DLC GUI directly, run

```bash
./deeplabcut-docker.sh gui
```

### Interactive console with DLC in light mode

```bash
./deeplabcut-docker.sh bash
```

### Jupyter Notebooks

```bash
./deeplabcut-docker.sh notebook
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
