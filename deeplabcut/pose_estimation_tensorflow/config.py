"""
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

import logging
import pprint

import yaml


def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    for k, v in a.items():
        # a must specify keys that are in b
        # if k not in b:
        #    raise KeyError('{} is not a valid config key'.format(k))

        # recursively merge dicts
        if isinstance(v, dict):
            if not b.get(k, False):
                b[k] = v
            else:
                try:
                    _merge_a_into_b(a[k], b[k])
                except:
                    print("Error under config key: {}".format(k))
                    raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config from file filename and merge it into the default options.
    """
    with open(filename, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Update the snapshot path to the corresponding path!
    trainpath = str(filename).split("pose_cfg.yaml")[0]
    yaml_cfg["snapshot_prefix"] = trainpath + "snapshot"
    # the default is: "./snapshot"

    # reloading defaults, as they can bleed over from a previous run otherwise
    import importlib
    from . import default_config

    importlib.reload(default_config)

    default_cfg = default_config.cfg
    _merge_a_into_b(yaml_cfg, default_cfg)

    logging.info("Config:\n" + pprint.pformat(default_cfg))
    return default_cfg  # updated


def load_config(filename="pose_cfg.yaml"):
    return cfg_from_file(filename)


if __name__ == "__main__":
    print(load_config())
