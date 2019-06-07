import os
os.environ["DLClight"]="True"
#os.environ["Colab"]="True"

import deeplabcut as dlc

config_path = 'home/tjv55/open_field-Tom-2019-04-09/config.yaml'

dlc.train_network(config_path)
