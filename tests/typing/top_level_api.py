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
"""Static smoke test for DeepLabCut's top-level public API.

This file is analyzed by a type checker (Pyright/basedpyright) rather than
executed by pytest. The ``reveal_type`` calls below must resolve to real
signatures — never ``Any`` or ``Unknown`` — which proves that
``deeplabcut/__init__.pyi`` exposes the lazy exports statically.
"""

import deeplabcut

reveal_type(deeplabcut.analyze_images)
reveal_type(deeplabcut.analyze_videos)
reveal_type(deeplabcut.train_network)
reveal_type(deeplabcut.evaluate_network)
reveal_type(deeplabcut.create_new_project)
reveal_type(deeplabcut.create_training_dataset)
reveal_type(deeplabcut.Engine)
reveal_type(deeplabcut.VERSION)
reveal_type(deeplabcut.DEBUG)
reveal_type(deeplabcut.launch_dlc)
reveal_type(deeplabcut.transformer_reID)
