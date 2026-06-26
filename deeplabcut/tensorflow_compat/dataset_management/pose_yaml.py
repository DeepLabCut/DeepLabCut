#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""TensorFlow pose configuration YAML helpers."""

from __future__ import annotations

import yaml


def ParseYaml(configfile):
    raw = open(configfile).read()
    docs = []
    for raw_doc in raw.split("\n---"):
        try:
            docs.append(yaml.load(raw_doc, Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)
    return docs


def MakeTrain_pose_yaml(
    itemstochange,
    saveasconfigfile,
    defaultconfigfile,
    items2drop: dict | None = None,
    save: bool = True,
):
    if items2drop is None:
        items2drop = {}

    docs = ParseYaml(defaultconfigfile)
    for key in items2drop.keys():
        if key in docs[0].keys():
            docs[0].pop(key)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    if save:
        with open(saveasconfigfile, "w") as f:
            yaml.dump(docs[0], f)

    return docs[0]


def MakeTest_pose_yaml(
    dictionary,
    keys2save,
    saveasfile,
    nmsradius=None,
    minconfidence=None,
    sigma=None,
    locref_smooth=None,
):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    # adding important values for multianiaml project:
    if nmsradius is not None:
        dict_test["nmsradius"] = nmsradius
    if minconfidence is not None:
        dict_test["minconfidence"] = minconfidence
    if sigma is not None:
        dict_test["sigma"] = sigma
    if locref_smooth is not None:
        dict_test["locref_smooth"] = locref_smooth

    dict_test["scoremap_dir"] = "test"
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)


def MakeInference_yaml(itemstochange, saveasconfigfile, defaultconfigfile):
    docs = ParseYaml(defaultconfigfile)
    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]
