# DeepLabCut Dockerfiles

This repo contains build routines for the following official DeepLabCut docker images:
- `deeplabcut/deeplabcut:latest-core`: TF2.5, cuDNN8 and DLC in light mode
- `deeplabcut/deeplabcut:latest-gui`: TF2.5, cuDNN8 and DLC in GUI mode

All images are based on Python 3.8.
The images are synced to dockerhub: https://hub.docker.com/r/deeplabcut/deeplabcut

## Quickstart

For a quick start, you can build a local image based on the DLC images by running

``` bash
docker/interact.sh [image name]
```

which will give you a [nice](https://ohmybash.nntoan.com/) shell with correct user permissions along with a `git` setup within the container.
This environment might be useful for prototyping. You'll have read access to the files in your current directory.

## For developers

Make sure your docker deamon is running and navigate to the repository root directory.
You can build the images by running

```
docker/build.sh
```

Note that this assumes that you have rights to execute `docker build` and `docker run` commands which requires either `sudo` access or membership in the `docker` group on your local machine. The script determines the correct mode automatically.

Built images can be pushed to dockerhub by running

```
docker/push.sh
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
