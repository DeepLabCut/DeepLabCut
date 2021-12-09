# DeepLabCut Dockerfiles

**Note that this README is mainly intended for DeepLabCut developers. The main documentation contains its own user documentation on the provided docker images.**

This repo contains build routines for the following official DeepLabCut docker images:
- `deeplabcut/deeplabcut:base`: Base image with TF2.5, cuDNN8 and DLC
- `deeplabcut/deeplabcut:latest-core`: DLC in light mode
- `deeplabcut/deeplabcut:latest-gui`: DLC in GUI mode
- `deeplabcut/deeplabcut:latest-gui-jupyter`: DLC in GUI mode, with jupyter installed

All images are based on Python 3.8.
The images are synced to DockerHub: https://hub.docker.com/r/deeplabcut/deeplabcut

## Quickstart

You can use the images fully standalone, without the need of cloning the DeepLabCut repo.
A helper package called `deeplabcut-docker` is available on PyPI and can be installed by running

``` bash
pip install deeplabcut-docker
```

*Note: Advanced users can also directly download and use the `deeplabcut-docker.sh` script if this is preferred over a python helper script.*

We provide docker containers for three different use cases outlined below.

In all cases, your current directory will be mounted in the container, and the container
will be started with your current username and group.

- To launch the DLC GUI directly, run
  ```bash
  deeplabcut-docker gui
  ```
- Interactive console with DLC in light mode
  ```bash
  deeplabcut-docker bash
  ```
- A Jupyter notebook server can be launched with
  ```bash
  deeplabcut-docker notebook
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
