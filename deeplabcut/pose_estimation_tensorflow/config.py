'''
Adopted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

'''

import pprint
import logging

import yaml
from easydict import EasyDict as edict

from . import default_config

cfg = default_config.cfg


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        #if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config from file filename and merge it into the default options.
    """
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f,Loader=yaml.SafeLoader))

    #Update the snapshot path to the corresponding path!
    trainpath=str(filename).split('pose_cfg.yaml')[0]
    yaml_cfg['snapshot_prefix']=trainpath+'snapshot'
    #the default is: "./snapshot"
    _merge_a_into_b(yaml_cfg, cfg)

    logging.info("Config:\n"+pprint.pformat(cfg))
    return cfg


def load_config(filename = "pose_cfg.yaml"):
    return cfg_from_file(filename)

if __name__ == "__main__":
    print(load_config())
