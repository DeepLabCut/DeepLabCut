#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Functional tests for MainWindow config caching and error recovery.

These construct a real (off-screen) MainWindow but stub out ``add_tabs`` and
the error dialog, so no DLC project on disk and no user interaction is needed.
"""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6 import QtWidgets

pytestmark = pytest.mark.functional


class TestCfgCaching:
    def test_cfg_is_cached_between_accesses(self, main_window, tmp_path, write_project_config):
        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path)

        main_window.config = str(config_path)

        first = main_window.cfg
        assert first is not None
        assert first is main_window.cfg  # same validated snapshot

    def test_external_edit_does_not_silently_reload(self, main_window, tmp_path, write_project_config):
        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path)
        main_window.config = str(config_path)
        first = main_window.cfg

        write_project_config(config_path, tmp_path, task="edited")

        # The GUI keeps operating on the loaded snapshot until an explicit reload.
        assert main_window.cfg is first
        assert main_window.cfg.Task == "demo"

    def test_assigning_config_invalidates_cache(self, main_window, tmp_path, write_project_config):
        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path)
        main_window.config = str(config_path)
        first = main_window.cfg

        write_project_config(config_path, tmp_path, task="edited")
        main_window.config = str(config_path)  # explicit reload boundary

        reloaded = main_window.cfg
        assert reloaded is not first
        assert reloaded.Task == "edited"

    def test_cfg_is_none_without_a_project(self, main_window):
        main_window.config = None
        assert main_window.cfg is None


class TestRecoveryLoop:
    """_build_project_ui_from_current_config must never crash the window."""

    @pytest.fixture
    def stubbed_window(self, main_window):
        main_window._built_tabs = []
        main_window.add_tabs = lambda: main_window._built_tabs.append(True)
        return main_window

    def test_cancel_leaves_welcome_page(self, stubbed_window, tmp_path, write_project_config):
        from deeplabcut.gui.window import ConfigErrorAction

        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path, extra="not_a_dlc_setting: 1")

        handled = []

        def cancel(error):
            handled.append(error)
            return ConfigErrorAction.CANCEL

        stubbed_window._handle_config_error = cancel
        stubbed_window.config = str(config_path)

        assert stubbed_window._build_project_ui_from_current_config() is False
        assert len(handled) == 1
        assert stubbed_window._built_tabs == []

    def test_retry_succeeds_after_user_fixes_config(self, stubbed_window, tmp_path, write_project_config):
        from deeplabcut.gui.window import ConfigErrorAction

        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path, extra="not_a_dlc_setting: 1")

        def fix_file_and_retry(error):
            write_project_config(config_path, tmp_path, task="repaired")
            return ConfigErrorAction.RETRY

        stubbed_window._handle_config_error = fix_file_and_retry
        stubbed_window.config = str(config_path)

        assert stubbed_window._build_project_ui_from_current_config() is True
        assert stubbed_window._built_tabs == [True]
        # Each retry re-reads from disk, so the repaired file is what got loaded.
        assert stubbed_window.cfg.Task == "repaired"

    def test_valid_config_builds_tabs_without_error_handling(self, stubbed_window, tmp_path, write_project_config):
        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path)

        def unexpected(error):  # pragma: no cover - should not run
            raise AssertionError(f"error handler should not be called: {error}")

        stubbed_window._handle_config_error = unexpected
        stubbed_window.config = str(config_path)

        assert stubbed_window._build_project_ui_from_current_config() is True
        assert stubbed_window._built_tabs == [True]


class TestConfigCacheInvalidation:
    def test_invalidate_drops_cache_for_next_access(self, main_window, tmp_path, write_project_config):
        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path)
        main_window.config = str(config_path)

        first = main_window.cfg
        assert first is not None

        write_project_config(config_path, tmp_path, task="changed-on-disk")
        main_window.invalidate_config_cache()
        reloaded = main_window.cfg

        assert reloaded is not first
        assert reloaded.Task == "changed-on-disk"

    def test_invalidate_is_idempotent(self, main_window):
        # Must not raise even with no config loaded.
        main_window.config = None
        main_window.invalidate_config_cache()
        assert main_window.cfg is None


class TestReloadTimer:
    def test_named_timer_triggers_reload(self, qapp, qtbot, tmp_path, write_project_config, monkeypatch):
        """Verify the _reload_timer in ManageProject fires reload_project_config."""
        from deeplabcut.gui.tabs.manage_project import ManageProject

        # Keep QSettings out of the real user settings.
        qapp.setOrganizationName("DeepLabCut-Tests")
        qapp.setApplicationName("DLC-GUI-Tests")
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
        )

        config_path = tmp_path / "config.yaml"
        write_project_config(config_path, tmp_path)

        from deeplabcut.gui.window import MainWindow

        window = MainWindow(qapp)
        try:
            window.config = str(config_path)

            reloads = []
            monkeypatch.setattr(
                window,
                "reload_project_config",
                lambda: reloads.append(True),
            )

            tab = ManageProject(window, window, "")
            tab._reload_timer.start()

            # The timer has interval=0 so it fires on the next event loop tick.
            qtbot.wait(50)

            assert len(reloads) == 1
        finally:
            window.close()
