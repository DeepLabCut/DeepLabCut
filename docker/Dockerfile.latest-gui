FROM deeplabcut/deeplabcut:base

RUN DEBIAN_FRONTEND=noninteractive apt-get update -yy \ 
    && DEBIAN_FRONTEND=noninteractive \
         apt-get install -yy --no-install-recommends \
         libgtk-3-dev python3-wxgtk4.0 locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8 en_GB.UTF-8

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --upgrade "deeplabcut[gui]>=2.2.0.2" numpy==1.19.5 decorator==4.4.2 tensorflow==2.5.0

ENV DLClight=False
CMD ["python3", "-m", "deeplabcut"]
