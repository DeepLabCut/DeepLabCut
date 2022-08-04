FROM deeplabcut/deeplabcut:latest-gui
RUN pip install --no-cache-dir jupyter
COPY ./examples /app/examples
CMD ["jupyter", "notebook", \
     "--no-browser", \
     "--NotebookApp.password", "sha3_224:019dc9326f7c:689d6c568f840f6126fb438ba0cdb8ce94ddd0769399e152caf79931", \
     "--ip", "0.0.0.0" ]
