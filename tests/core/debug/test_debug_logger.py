from __future__ import annotations

import logging
from uuid import uuid4

import pytest

from deeplabcut.core.debug import (
    InMemoryDebugRecorder,
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
    recorder = install_debug_recorder(logger_name=logger_name, capacity=10)

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
    recorder = install_debug_recorder(logger_name=logger_name, capacity=2)

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
