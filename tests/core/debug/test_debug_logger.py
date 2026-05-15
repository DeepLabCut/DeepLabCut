from __future__ import annotations

import logging
from uuid import uuid4

import pytest

import deeplabcut.core.debug.debug_logger as debug_mod
from deeplabcut.core.debug import (
    DebugSection,
    ExecutableSpec,
    InMemoryDebugRecorder,
    LibrarySpec,
    build_debug_report,
    collect_executable_summary,
    collect_version_summary,
    format_debug_report,
    get_debug_recorder,
    install_debug_recorder,
    log_timing,
)


@pytest.fixture
def logger_name() -> str:
    return f"deeplabcut.tests.debug.{uuid4()}"


@pytest.fixture
def clean_logger(logger_name: str):
    """Create an isolated logger namespace and fully clean it afterwards."""
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    old_propagate = logger.propagate
    old_handlers = list(logger.handlers)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    yield logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    for handler in old_handlers:
        logger.addHandler(handler)

    logger.setLevel(old_level)
    logger.propagate = old_propagate

    # Remove recorder marker installed by install_debug_recorder().
    logger.__dict__.pop("_dlc_debug_recorder", None)


def test_install_debug_recorder_is_idempotent(logger_name: str, clean_logger):
    recorder1 = install_debug_recorder(logger_name=logger_name, capacity=10)
    recorder2 = install_debug_recorder(logger_name=logger_name, capacity=99)

    assert recorder1 is recorder2
    assert isinstance(recorder1, InMemoryDebugRecorder)
    assert get_debug_recorder(logger_name=logger_name) is recorder1


def test_recorder_captures_messages_and_exceptions(logger_name: str, clean_logger):
    logger = clean_logger
    recorder = install_debug_recorder(logger_name=logger_name, capacity=10, handler_level=logging.DEBUG)

    logger.info("hello %s", "dlc")
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("something failed")

    records = recorder.snapshot()

    assert len(records) == 2
    assert records[0].message == "hello dlc"
    assert records[0].level == "INFO"
    assert records[1].message == "something failed"
    assert records[1].level == "ERROR"
    assert records[1].exc_text is not None
    assert "ValueError: boom" in records[1].exc_text


def test_recorder_is_bounded(logger_name: str, clean_logger):
    logger = clean_logger
    recorder = install_debug_recorder(logger_name=logger_name, capacity=2, handler_level=logging.DEBUG)

    logger.debug("first")
    logger.debug("second")
    logger.debug("third")

    messages = [rec.message for rec in recorder.snapshot()]
    assert messages == ["second", "third"]


def test_render_text_contains_recent_messages(logger_name: str, clean_logger):
    logger = clean_logger
    recorder = install_debug_recorder(logger_name=logger_name, capacity=5)

    logger.warning("alpha")
    logger.error("beta")

    text = recorder.render_text(limit=10)

    assert "WARNING" in text
    assert "ERROR" in text
    assert "alpha" in text
    assert "beta" in text
    assert logger_name in text


def test_clear_resets_records_and_drop_count(logger_name: str, clean_logger):
    logger = clean_logger
    recorder = install_debug_recorder(logger_name=logger_name, capacity=5)

    logger.info("before clear")
    assert recorder.snapshot()

    recorder.clear()

    assert recorder.snapshot() == []
    assert recorder.dropped_count == 0
    assert recorder.render_text() == ""


def test_log_timing_emits_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    logger_name: str,
    clean_logger,
):
    logger = clean_logger
    calls: list[tuple[int, str, tuple[object, ...]]] = []

    wrapped = log_timing.__wrapped__

    monkeypatch.setitem(wrapped.__globals__, "DLC_LOG_TIMING", True)

    ticks = iter([1_000_000_000, 1_005_000_000])  # 5.000 ms
    monkeypatch.setitem(wrapped.__globals__, "perf_counter_ns", lambda: next(ticks))

    monkeypatch.setattr(logger, "isEnabledFor", lambda level: True)

    def fake_log(level, msg, *args):
        calls.append((level, msg, args))

    monkeypatch.setattr(logger, "log", fake_log)

    with log_timing(logger, "tiny-step", threshold_ms=0.0):
        pass

    assert calls == [
        (logging.DEBUG, "%s took %.3f ms", ("tiny-step", 5.0)),
    ]


def test_log_timing_is_silent_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    logger_name: str,
    clean_logger,
):
    logger = clean_logger
    calls: list[tuple[int, str, tuple[object, ...]]] = []

    wrapped = log_timing.__wrapped__

    monkeypatch.setitem(wrapped.__globals__, "DLC_LOG_TIMING", False)
    monkeypatch.setattr(logger, "isEnabledFor", lambda level: True)

    def fake_log(level, msg, *args):
        calls.append((level, msg, args))

    monkeypatch.setattr(logger, "log", fake_log)

    with log_timing(logger, "should-not-appear", threshold_ms=0.0):
        pass

    assert calls == []


def test_log_timing_respects_threshold(
    monkeypatch: pytest.MonkeyPatch,
    logger_name: str,
    clean_logger,
):
    logger = clean_logger
    calls: list[tuple[int, str, tuple[object, ...]]] = []

    wrapped = log_timing.__wrapped__

    monkeypatch.setitem(wrapped.__globals__, "DLC_LOG_TIMING", True)

    ticks = iter([1_000_000_000, 1_001_000_000])  # 1.000 ms
    monkeypatch.setitem(wrapped.__globals__, "perf_counter_ns", lambda: next(ticks))

    monkeypatch.setattr(logger, "isEnabledFor", lambda level: True)

    def fake_log(level, msg, *args):
        calls.append((level, msg, args))

    monkeypatch.setattr(logger, "log", fake_log)

    with log_timing(logger, "tiny-step", threshold_ms=2.0):
        pass

    assert calls == []


# ----------- Report building tests -----------
def test_build_debug_report_includes_runtime_libraries_tools_and_recent_logs(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = InMemoryDebugRecorder(capacity=10, level=logging.DEBUG)

    record = logging.LogRecord(
        name="deeplabcut.tests.debug",
        level=logging.INFO,
        pathname=__file__,
        lineno=123,
        msg="hello %s",
        args=("report",),
        exc_info=None,
    )
    recorder.handle(record)

    monkeypatch.setattr(
        debug_mod,
        "collect_runtime_summary",
        lambda: {
            "python": "3.11.9",
            "platform": "TestOS-1.0",
            "executable": "bin/python",
        },
    )

    monkeypatch.setattr(
        debug_mod,
        "_version",
        lambda dist_name: {
            "alpha": "1.2.3",
            "opencv-python": "9.9.9-dist",
        }.get(dist_name, "not-installed"),
    )

    monkeypatch.setattr(
        debug_mod,
        "_module_version",
        lambda module_name: {
            "cv2": "4.10.0",
        }.get(module_name, "not-installed"),
    )

    monkeypatch.setattr(
        debug_mod,
        "_module_path",
        lambda module_name: {
            "alpha": "/tmp/site-packages/alpha/__init__.py",
            "cv2": "/tmp/site-packages/cv2/__init__.py",
        }.get(module_name, "unknown"),
    )

    monkeypatch.setattr(
        debug_mod,
        "_command_version",
        lambda command, version_args: {
            "ffmpeg": "ffmpeg 6.1",
        }.get(command, "unavailable"),
    )

    monkeypatch.setattr(
        debug_mod,
        "_which",
        lambda command: {
            "ffmpeg": "/usr/bin/ffmpeg",
        }.get(command, "not-found"),
    )

    report = build_debug_report(
        recorder=recorder,
        libraries=(
            LibrarySpec("alpha"),
            LibrarySpec(
                "opencv-python",
                dist_name="opencv-python",
                module_name="cv2",
                prefer_module_version=True,
            ),
        ),
        executables=(ExecutableSpec("ffmpeg"),),
        include_module_paths=True,
        include_executable_paths=True,
        log_limit=20,
    )

    assert "## Runtime" in report
    assert "- python: 3.11.9" in report
    assert "- platform: TestOS-1.0" in report
    assert "- executable: bin/python" in report

    assert "## Libraries" in report
    assert "- alpha: 1.2.3" in report
    assert "- opencv-python: 4.10.0" in report
    assert "- alpha_module_path: alpha/__init__.py" in report
    assert "- opencv-python_module_path: cv2/__init__.py" in report

    assert "## External tools" in report
    assert "- ffmpeg: ffmpeg 6.1" in report
    assert "- ffmpeg_path: bin/ffmpeg" in report

    assert "## Recent logs" in report
    assert "deeplabcut.tests.debug" in report
    assert "INFO" in report
    assert "hello report" in report
    assert "```text" in report


def test_build_debug_report_default_grouped_sections_and_skips_unavailable_tf(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        debug_mod,
        "collect_runtime_summary",
        lambda: {
            "python": "3.12.0",
            "platform": "GroupedTestOS",
            "executable": "python",
        },
    )

    def fake_collect_version_summary(*, libraries=None, include_module_paths=False):
        if libraries == debug_mod.DLC_CORE_LIBS:
            return {"deeplabcut": "1.0.0", "numpy": "2.0.0"}
        if libraries == debug_mod.DLC_GUI_LIBS:
            return {"PySide6": "6.8.0"}
        if libraries == debug_mod.DLC_TF_LIBS:
            return {
                "tensorflow": "not-installed",
                "tf_keras": "not-installed",
                "tensorpack": "unknown",
                "tf_slim": "not-installed",
            }
        raise AssertionError("unexpected libraries input")

    monkeypatch.setattr(debug_mod, "collect_version_summary", fake_collect_version_summary)

    monkeypatch.setattr(
        debug_mod,
        "collect_executable_summary",
        lambda *, executables=None, include_paths=True: {
            "ffmpeg": "unavailable",
            "ffmpeg_path": "not-found",
        },
    )

    report = build_debug_report(
        recorder=None,
        libraries=None,
        executables=None,
    )

    assert "## Runtime" in report
    assert "## DeepLabCut core libraries" in report
    assert "- deeplabcut: 1.0.0" in report
    assert "## GUI libraries" in report
    assert "- PySide6: 6.8.0" in report

    # All TF values are unavailable/unknown, so the section should be omitted.
    assert "## TensorFlow libraries" not in report

    # External tools should still be shown even when unavailable.
    assert "## External tools" in report
    assert "- ffmpeg: unavailable" in report
    assert "- ffmpeg_path: not-found" in report

    assert "## Recent logs" in report
    assert "<debug recorder unavailable>" in report


def test_collect_version_summary_prefers_module_version_and_falls_back_to_distribution(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        debug_mod,
        "_module_version",
        lambda module_name: {
            "cv2": "not-installed",
        }.get(module_name, "unknown"),
    )

    monkeypatch.setattr(
        debug_mod,
        "_version",
        lambda dist_name: {
            "opencv-python": "4.9.0.80",
            "missing-lib": "not-installed",
        }.get(dist_name, "not-installed"),
    )

    summary = collect_version_summary(
        libraries=(
            LibrarySpec(
                "opencv-python",
                dist_name="opencv-python",
                module_name="cv2",
                prefer_module_version=True,
            ),
            LibrarySpec("missing-lib"),
        )
    )

    assert summary["opencv-python"] == "4.9.0.80"
    assert summary["missing-lib"] == "not-installed"


def test_collect_executable_summary_reports_unavailable_tool_and_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(debug_mod, "_command_version", lambda command, version_args: "unavailable")
    monkeypatch.setattr(debug_mod, "_which", lambda command: "not-found")

    summary = collect_executable_summary(
        executables=(ExecutableSpec("ghosttool"),),
        include_paths=True,
    )

    assert summary == {
        "ghosttool": "unavailable",
        "ghosttool_path": "not-found",
    }


def test_build_debug_report_uses_no_captured_logs_placeholder_for_empty_recorder(
    monkeypatch: pytest.MonkeyPatch,
):
    recorder = InMemoryDebugRecorder(capacity=5, level=logging.DEBUG)

    monkeypatch.setattr(
        debug_mod,
        "collect_runtime_summary",
        lambda: {
            "python": "3.11.0",
            "platform": "EmptyLogsOS",
            "executable": "python",
        },
    )

    monkeypatch.setattr(
        debug_mod,
        "collect_executable_summary",
        lambda *, executables=None, include_paths=True: {},
    )

    report = build_debug_report(
        recorder=recorder,
        libraries=(),
        executables=(),
    )

    assert "## Runtime" in report
    assert "## Libraries" in report
    assert "- <no data>" in report
    assert "## Recent logs" in report
    assert "<no captured logs>" in report


def test_format_debug_report_renders_empty_section_and_logs_block():
    text = format_debug_report(
        sections=[
            DebugSection(title="Example", items={}),
        ],
        logs_text="line one\nline two",
    )

    assert "## Example" in text
    assert "- <no data>" in text
    assert "## Recent logs" in text
    assert "```text" in text
    assert "line one" in text
    assert "line two" in text
