# DeepLabCut Docker containers

For DeepLabCut 2.2.0.2 and onwards, we provide container containers on [DockerHub](https://hub.docker.com/r/deeplabcut/deeplabcut). Using Docker is an alternative approach to using DeepLabCut, which only requires the user to install [Docker](https://www.docker.com/) on your machine, vs. following the step-by-step installation guide for a Anaconda setup. All dependencies needed to run DeepLabCut in terminal or GUI mode, or running Jupyter notebooks with DeepLabCut pre-installed are shipped with the provided Docker images.

Advanced users can directly head to [DockerHub](https://hub.docker.com/r/deeplabcut/deeplabcut) and use the provided images there. To get started with using the images, we however also provide a helper tool, `deeplabcut-docker`, which makes the transition to docker images particularly convenient; to install the tool, run

``` bash
$ pip install deeplabcut-docker
```

on your machine (potentially in a virtual environment, or an existing Anaconda environment).
Note that this will *not* disprupt or install Tensorflow, or any other DeepLabCut dependencies on your computer---the Docker containers are completely isolated from your existing software installation!

## Usage modes

With `deeplabcut-docker`, you can use the images in three modes.

- *Note 1: When running any of the following commands first, it can take some time to complete (a few minutes, depending on your internet connection), since it downloads the Docker image in the background. If you do not see any errors in your terminal, assume that everything is working fine! Subsequent runs of the command will be faster.*
- *Note 2: The Terminal mode image can be used on all supported platforms (Linux and MacOS). The GUI images can only be considered stable on Linux systems as of DeepLabCut 2.2.0.2 and need additional configuration on Mac.*
- *Note 3: For any mode below, you might want to set which directory is the base, namely, so you can have read/write (or ready only access). Here is how to do so:
If you want to mount the whole directory could e.g., pass*

`deeplabcut-docker bash -v /home/mackenzie/DEEPLABCUT:/home/mackenzie/DEEPLABCUT`

(which will mount the full directory into the container in read/write mode)

If read-only access is enough, `deeplabcut-docker bash -v /home/mackenzie/DEEPLABCUT:/home/mackenzie/DEEPLABCUT:ro`


### GUI mode

To run DeepLabCut in GUI mode and start the DeepLabCut GUI, simply run

``` bash
$ deeplabcut-docker gui
```

which will pull the latest DeepLabCut version along with all dependencies, and afterwards opens the DeepLabCut GUI.

The DeepLabCut version in this container is equivalent to the one you install with `pip install "deeplabcut[gui]"`.

### Terminal mode 

If you not need the GUI, you can run the light version of DeepLabCut and open a terminal by running

``` bash
$ deeplabcut-docker bash
```

Inside the terminal, you can confirm that DeepLabCut is correctly installed by running

``` bash
$ ipython
>>> import deeplabcut
>>> print(deeplabcut.__version__)
2.2.0.2
```

### Jupyter Notebook mode

Finally, you can run DeepLabCut by starting a jupyter notebook server. The corresponding image can be pulled and started by running

``` bash
$ deeplabcut-docker notebook 
```

which will start a Jupyter notebook server. Follow the terminal instructions to open the notebook, by entering `http://127.0.0.1:8888` in your favorite browser. When prompted for a password, use `deeplabcut`, which is the pre-set option in the container.

The DeepLabCut version in this container is equivalent to the one you install with `pip install deeplabcut[gui]`. This means that you can start the DeepLabCut GUI with the appropriate commands in your notebook!

### Advanced usage

Advanced users and developers can visit the `/docker` subdirectory in the DeepLabCut codebase on Github. We provide Dockerfiles for all images, along with build instructions there.

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


## Notes and troubleshooting

Running Docker with GUI support can vary across systems. The current images are confirmed to work with different Linux systems, but especially on MacOS additional configuration steps are necessary.

When running containers on Linux, in some systems it might be necessary to run `host +local:docker` before starting the image via `deeplabcut-docker`.

If you encounter errors while using the images, please open an issue in the DeepLabCut repo---especially the `deeplabcut-docker` is still in its alpha version, and we appreciate user feedback to make the tool robust to use across many operating systems!
