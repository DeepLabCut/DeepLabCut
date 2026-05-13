#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

from collections.abc import Sequence

from .debug_dialog import (
    DebugTextDialog,
    create_generate_debug_log_action,
    make_issue_report_provider,
    make_log_text_provider,
    show_debug_report_dialog,
)

__all__: Sequence[str] = (
    "DebugTextDialog",
    "create_generate_debug_log_action",
    "make_issue_report_provider",
    "make_log_text_provider",
    "show_debug_report_dialog",
)
