---
deeplabcut:
  last_content_updated: '2026-05-22'
  last_metadata_updated: '2026-05-22'
  ignore: false
  visibility: online
  status: viable
  last_verified: '2026-05-22'
  verified_for: 3.0.0
---

(docker-containers)=

# DeepLabCut in Docker

From DeepLabCut 2.2.0.2 onward, we provide container images on [DockerHub](https://hub.docker.com/r/deeplabcut/deeplabcut).
Using Docker is an alternative approach to installing DeepLabCut in a local conda or pip environment: the images bundle all dependencies needed to run DeepLabCut in a reproducible, self-contained environment.
In a Docker container, DeepLabCut can be used from the terminal, or with Jupyter notebooks; the DeepLabCut GUI is not supported.
The approach requires a local installation of [Docker / Docker Desktop](https://www.docker.com/), and is meant for users who need strict reproducibility, an isolated environment, or server-based automation.

```{important}
The napari-deeplabcut plugin **cannot be run in a Docker container**. To label
your data, please {ref}`install napari-deeplabcut <file:napari-gui-landing>` in a local, non-dockerized environment, e.g. using pip: `pip install napari-deeplabcut` .
```

Advanced users can directly head to [DockerHub](https://hub.docker.com/r/deeplabcut/deeplabcut) and use the provided images there. To get started with using the images, we however also provide a helper tool, `deeplabcut-docker`, which makes the transition to docker images particularly convenient; to install the tool, run

```bash
$ pip install deeplabcut-docker
```

on your machine (in any environment). `deeplabcut-docker` is just a lightweight package for setting up the Docker environment and it will *not* disrupt your existing software installation. The Docker container itself is completely isolated from your local environment!

## Available images

The following images are published to [DockerHub](https://hub.docker.com/r/deeplabcut/deeplabcut). All images come with Python 3.11 and CUDA pre-installed.

| Tag                                                  | Description                            |
| ---------------------------------------------------- | -------------------------------------- |
| `deeplabcut/deeplabcut:latest`                       | Default runtime image for terminal use |
| `deeplabcut/deeplabcut:latest-jupyter`               | Jupyter Notebook server                |
| `deeplabcut/deeplabcut:<VERSION>-core-cuda<CUDA>`    | Versioned runtime image                |
| `deeplabcut/deeplabcut:<VERSION>-jupyter-cuda<CUDA>` | Versioned Jupyter image                |

By default `deeplabcut-docker` pulls the `latest` / `latest-jupyter` tag. To select a specific DeepLabCut or CUDA version, set the `DLC_VERSION` and `CUDA_VERSION` environment variables:

```bash
DLC_VERSION=3.0.0 CUDA_VERSION=12.4 deeplabcut-docker bash --gpus all
```

To use a completely custom image instead of the default tags, pass `--image repo:tag`. Make sure the image supports Jupyter notebooks when using `deeplabcut-docker notebook`.

## Usage modes

With `deeplabcut-docker`, you can use the images in two modes.

<!-- - *Note 2: The labelling GUI cannot be used through the Docker images. However, you can install [`napari-deeplabcut`](https://github.com/DeepLabCut/napari-deeplabcut/tree/main?tab=readme-ov-file#napari-deeplabcut-keypoint-annotation-for-pose-estimation) in a conda environment to do the labelling!* -->

```{note}
1. When running any of the following commands first, it can take some time to complete (a few minutes, depending on your internet connection), since it downloads the Docker image in the background. If you do not see any errors in your terminal, assume that everything is working fine! Subsequent runs of the command will be faster.
2. For any mode below, you might want to set which directory is the base, namely, so you can have read/write (or read-only access). Here is how to do so:
  If you want to mount the whole directory could e.g., pass
  `deeplabcut-docker bash -v /home/mackenzie/DEEPLABCUT:/home/mackenzie/DEEPLABCUT`
  (which will mount the full directory into the container in read/write mode)
  If read-only access is enough, `deeplabcut-docker bash -v /home/mackenzie/DEEPLABCUT:/home/mackenzie/DEEPLABCUT:ro`
```

### Terminal mode

You can run the light version of DeepLabCut and open a terminal by running

```bash
$ deeplabcut-docker bash
```

````{important}
If you have GPUs on your machine and want to use them to train models, you
need to pass the `--gpus all` argument to `deeplabcut-docker`:

```bash
$ deeplabcut-docker bash --gpus all
```
````

Inside the terminal, you can confirm that DeepLabCut is correctly installed by running and noting which version installs.

```bash
$ ipython
>>> import deeplabcut
```

### Jupyter Notebook mode

You can run DeepLabCut by starting a jupyter notebook server. The corresponding image can be pulled and started by running

```bash
$ deeplabcut-docker notebook
```

which will start a Jupyter notebook server. Follow the terminal instructions to open the notebook, by entering `http://127.0.0.1:8888` in your favorite browser. When prompted for a password, use `deeplabcut`, which is the pre-set option in the container.

The container comes with `deeplabcut[modelzoo,wandb]` pre-installed. Note that the DeepLabCut GUI is not available inside the container.

```{warning}
The Jupyter image uses a fixed default access token (`deeplabcut`) that is publicly known.
Anyone who can reach port 8888 on your machine can execute arbitrary code in the container.
Do not expose port 8888 to the internet (e.g. via a cloud VM's firewall or a public `0.0.0.0`
binding without a reverse proxy).
For local use, bind the port to localhost only (e.g. `-p 127.0.0.1:8888:8888`) and use SSH
port forwarding to access the server remotely (see below).
To use a custom token, pass `-e NOTEBOOK_TOKEN=<your-token>` to `docker run`.
```

### Jupyter Notebooks on remote servers

Sometimes you want to run Jupyter Notebooks on a remote server and connect from your local
browser. This requires SSH port forwarding. For general guidance see
[this StackOverflow post](https://stackoverflow.com/a/69244262) or the
[Jupyter Notebook docs](https://jupyter-notebook.readthedocs.io/en/4.x/public_server.html).

With `deeplabcut-docker` and `DLC_NOTEBOOK_PORT`, this is straightforward:

```bash
# Example: remote port XXXX=8889, local port YYYY=8890

# 1. Connect to your server with port forwarding
ssh -L localhost:8890:localhost:8889 you@your-server

# 2. On the remote server, launch the container
DLC_NOTEBOOK_PORT=8889 deeplabcut-docker notebook --gpus all

# 3. Open http://127.0.0.1:8890 in your local browser
```

### Advanced usage

Advanced users and developers can visit the [`/docker` subdirectory](https://github.com/DeepLabCut/DeepLabCut/tree/main/docker) in the DeepLabCut codebase on GitHub. It contains a single multi-stage Dockerfile covering all images, along with build instructions.

## Prerequisites (if you don't have Docker installed already)

**(1)** Install Docker. See https://docs.docker.com/install/ & for Ubuntu: https://docs.docker.com/install/linux/docker-ce/ubuntu/
Test docker:

```
$ sudo docker run hello-world
```

The output should be: `Hello from Docker! This message shows that your installation appears to be working correctly.`

\*if you get the error `docker: Error response from daemon: Unknown runtime specified nvidia.` just simply restart docker:

```
   $ sudo systemctl daemon-reload
   $ sudo systemctl restart docker
```

**(2)** Add your user to the docker group (https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user)
Quick guide to create the docker group and add your user:
Create the docker group.

```
$ sudo groupadd docker
```

Add your user to the docker group.

```
$ sudo usermod -aG docker $USER
```

(perhaps restart your computer (best) or (at min) open a new terminal to make sure that you are added from now on)

## Notes and troubleshooting

We dropped GUI support in 2.3.5+ due to too many numerous issues supporting them. Also please note these are tested on unix systems.

When running containers on Linux, in some systems it might be necessary to run `host +local:docker` before starting the image via `deeplabcut-docker`.

If you encounter errors while using the images, please open an issue in the DeepLabCut repo. We appreciate user feedback to make the tool robust across many operating systems!
