import deeplabcut

config = "/Users/alex/Code/dlc_playingdata/MultiMouse-Daniel-2019-12-16/config.yaml"
directory = "/Users/alex//Code/dlc_playingdata/videocompressed0"

# deeplabcut.evaluate_network(config,Shuffles=[0],plotting= "individual")

deeplabcut.evaluate_network(
    config, Shuffles=[0], plotting="individual", directory=directory
)
