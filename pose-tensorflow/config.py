import os
import pprint
import logging

import yaml
from easydict import EasyDict as edict

import default_config


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
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, cfg)

    logging.info("Config:\n"+pprint.pformat(cfg))
    return cfg


def load_config(filename = "pose_cfg.yaml"):
    if 'POSE_PARAM_PATH' in os.environ:
        filename = os.environ['POSE_PARAM_PATH'] + '/' + filename
    return cfg_from_file(filename)


if __name__ == "__main__":
    print(load_config())