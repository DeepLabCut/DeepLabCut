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

from __future__ import annotations

import logging
import platform
import sys
import threading
import traceback
from collections import deque
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from time import perf_counter_ns

from ._debug_utils import (
    _command_version,
    _env_flag,
    _env_optional_float,
    _which,
)

_DEBUG_HANDLER_ATTR = "_dlc_debug_recorder"
LOG_QUEUE_MAXLEN = 1000

# NOTE @C-Achard 2026-05-13: we may want to centralize env vars in a config/settings module in the future
DLC_LOG_TIMING = _env_flag("DLC_LOG_TIMING", default=False)
DLC_LOG_TIMING_THRESHOLD_MS = _env_optional_float("DLC_LOG_TIMING_THRESHOLD_MS", default=None)


@contextmanager
def log_timing(
    logger: logging.Logger,
    label: str,
    *,
    level: int = logging.DEBUG,
    threshold_ms: float | None = None,
):
    """Lightweight scoped timer for debug instrumentation.

    Uses perf_counter_ns() for monotonic timing.
    Logs only if logger is enabled for the requested level.
    Optionally suppresses tiny timings below ``threshold_ms``.
    """
    if not logger.isEnabledFor(level) or not DLC_LOG_TIMING:
        yield
        return

    effective_threshold_ms = threshold_ms if threshold_ms is not None else DLC_LOG_TIMING_THRESHOLD_MS
    t0 = perf_counter_ns()
    try:
        yield
    finally:
        dt_ms = (perf_counter_ns() - t0) / 1e6
        if effective_threshold_ms is None or dt_ms >= effective_threshold_ms:
            logger.log(level, "%s took %.3f ms", label, dt_ms)


@dataclass(frozen=True)
class RecordedLog:
    created: float
    level: str
    logger_name: str
    message: str
    exc_text: str | None = None


class InMemoryDebugRecorder(logging.Handler):
    """Lightweight, fail-open in-memory log recorder.

    Safety properties:
    - bounded memory via deque(maxlen=...)
    - no file/network I/O
    - swallow-all-errors in emit()
    - does not log from inside itself
    - stores only small text snapshots
    """

    def __init__(self, *, capacity: int = LOG_QUEUE_MAXLEN, level: int = logging.DEBUG):
        super().__init__(level=level)
        self._records: deque[RecordedLog] = deque(maxlen=max(1, int(capacity)))
        self._lock = threading.Lock()
        self._dropped = 0

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Never call logging from here.
            # Never inspect application objects.
            msg = self._safe_message(record)
            exc_text = self._safe_exception_text(record)

            snap = RecordedLog(
                created=float(getattr(record, "created", 0.0) or 0.0),
                level=str(getattr(record, "levelname", "UNKNOWN")),
                logger_name=str(getattr(record, "name", "")),
                message=msg,
                exc_text=exc_text,
            )

            with self._lock:
                self._records.append(snap)

        except Exception:
            # Fail open: never let diagnostics interfere with runtime behavior.
            try:
                self._dropped += 1
            except Exception:
                pass

    def clear(self) -> None:
        try:
            with self._lock:
                self._records.clear()
                self._dropped = 0
        except Exception:
            pass

    def snapshot(self) -> list[RecordedLog]:
        try:
            with self._lock:
                return list(self._records)
        except Exception:
            return []

    def render_text(self, *, limit: int = 200) -> str:
        lines: list[str] = []
        try:
            records = self.snapshot()[-max(1, int(limit)) :]
            if not records:
                if self._dropped:
                    return f"[debug-recorder] no captured logs, {self._dropped} internal failures"
                return ""

            base = records[0].created
            for rec in records:
                ts = datetime.fromtimestamp(rec.created).strftime("%H:%M:%S.%f")[:-3]
                if DLC_LOG_TIMING:
                    rel_ms = (rec.created - base) * 1000.0
                    lines.append(f"{ts} (+{rel_ms:8.1f} ms) | {rec.level:<8} | {rec.logger_name} | {rec.message}")
                else:
                    lines.append(f"{ts} | {rec.level:<8} | {rec.logger_name} | {rec.message}")
                if rec.exc_text:
                    lines.append(rec.exc_text.rstrip())

            if self._dropped:
                lines.append(f"[debug-recorder] dropped internal failures: {self._dropped}")
        except Exception:
            return "[debug-recorder] failed to render logs"
        return "\n".join(lines)

    @staticmethod
    def _safe_message(record: logging.LogRecord) -> str:
        try:
            return record.getMessage()
        except Exception:
            try:
                return str(record.msg)
            except Exception:
                return "<unrenderable log message>"

    @staticmethod
    def _safe_exception_text(record: logging.LogRecord) -> str | None:
        try:
            if not record.exc_info:
                return None
            return "".join(traceback.format_exception(*record.exc_info))
        except Exception:
            return "<exception details unavailable>"


@dataclass(frozen=True)
class DebugSection:
    title: str
    items: dict[str, str]


def install_debug_recorder(
    *,
    logger_name: str = "deeplabcut",
    capacity: int = LOG_QUEUE_MAXLEN,
    handler_level: int = logging.INFO,
    ensure_logger_level: int | None = None,
) -> InMemoryDebugRecorder:
    """Attach a single in-memory recorder to the requested logger namespace.

    Idempotent: repeated calls return the same recorder.
    """
    root_logger = logging.getLogger(logger_name)

    existing = getattr(root_logger, _DEBUG_HANDLER_ATTR, None)
    if isinstance(existing, InMemoryDebugRecorder):
        return existing

    recorder = InMemoryDebugRecorder(capacity=capacity, level=handler_level)
    recorder.set_name("deeplabcut-debug-recorder")

    # Important:
    # - attach only to a DLC-owned logger namespace, not the global root logger
    # - set logger level to DEBUG so DLC debug calls are emitted
    # - keep propagation unchanged
    root_logger.addHandler(recorder)

    if ensure_logger_level is not None:
        # Only lower verbosity if explicitly requested.
        if root_logger.getEffectiveLevel() > ensure_logger_level:
            root_logger.setLevel(ensure_logger_level)

    setattr(root_logger, _DEBUG_HANDLER_ATTR, recorder)
    return recorder


def get_debug_recorder(*, logger_name: str = "deeplabcut") -> InMemoryDebugRecorder | None:
    logger = logging.getLogger(logger_name)
    recorder = getattr(logger, _DEBUG_HANDLER_ATTR, None)
    return recorder if isinstance(recorder, InMemoryDebugRecorder) else None


# --------------------------
# Environment / version info
# --------------------------


@dataclass(frozen=True)
class LibrarySpec:
    """Small description of a library to report.

    Parameters
    ----------
    key:
        Label used in the output report.
    dist_name:
        Distribution name used by ``importlib.metadata.version``.
    module_name:
        Importable module name used to resolve a module path.
    """

    key: str
    dist_name: str | None = None
    module_name: str | None = None

    def resolved_dist_name(self) -> str:
        return self.dist_name or self.key

    def resolved_module_name(self) -> str:
        return self.module_name or self.key


DLC_CORE_LIBS: tuple[LibrarySpec, ...] = (
    LibrarySpec("deeplabcut"),
    LibrarySpec("torch"),
    LibrarySpec("torchvision"),
    LibrarySpec("numpy"),
    LibrarySpec("pandas"),
    LibrarySpec("scipy"),
    LibrarySpec("h5py"),
    LibrarySpec("tables"),
    LibrarySpec("opencv-python", dist_name="opencv-python", module_name="cv2"),
)
DLC_GUI_LIBS: tuple[LibrarySpec, ...] = (
    LibrarySpec("PySide6"),
    LibrarySpec("shiboken6"),
    LibrarySpec("qtpy", dist_name="QtPy"),
    LibrarySpec("qdarkstyle"),
    LibrarySpec("napari"),
    LibrarySpec("napari-deeplabcut", dist_name="napari-deeplabcut", module_name="napari_deeplabcut"),
)
DLC_TF_LIBS: tuple[LibrarySpec, ...] = (
    LibrarySpec("tensorflow"),
    LibrarySpec("tf_keras", dist_name="tf-keras"),
    LibrarySpec("tensorpack"),
    LibrarySpec("tf_slim", dist_name="tf-slim"),
)
DLC_ALL_LIBS_SPECS: tuple[LibrarySpec, ...] = DLC_CORE_LIBS + DLC_GUI_LIBS + DLC_TF_LIBS


def _normalize_library_specs(
    libraries: Iterable[LibrarySpec | str] | None,
) -> tuple[LibrarySpec, ...]:
    if libraries is None:
        return DLC_ALL_LIBS_SPECS

    normalized: list[LibrarySpec] = []
    for item in libraries:
        if isinstance(item, LibrarySpec):
            normalized.append(item)
        else:
            normalized.append(LibrarySpec(str(item)))
    return tuple(normalized)


def _version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except Exception:
        return "not-installed"


def _module_path(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        p = getattr(mod, "__file__", None)
        return str(Path(p).resolve()) if p else "unknown"
    except Exception:
        return "unknown"


def _safe_tail(pathlike: object) -> str:
    """Redact user-specific absolute paths.

    Keeps only the last 2 path components when possible.
    """
    try:
        p = Path(str(pathlike))
        parts = p.parts
        if len(parts) >= 2:
            return str(Path(*parts[-2:]).as_posix())
        return str(p.as_posix())
    except Exception:
        return str(pathlike)


def collect_version_summary(
    *,
    libraries: Iterable[LibrarySpec | str] | None = None,
    include_module_paths: bool = False,
) -> dict[str, str]:
    """Collect package versions for a configurable library list.

    The ``libraries`` argument is intentionally lightweight:
    - pass ``None`` to use ``DLC_ALL_LIBS_SPECS``
    - pass a list of strings for simple cases
    - pass ``LibrarySpec`` objects when distribution/module names differ
    """
    specs = _normalize_library_specs(libraries)
    summary: dict[str, str] = {}

    for spec in specs:
        key = spec.key
        summary[key] = _version(spec.resolved_dist_name())
        if include_module_paths:
            summary[f"{key}_module_path"] = _safe_tail(_module_path(spec.resolved_module_name()))

    return summary


@dataclass(frozen=True)
class ExecutableSpec:
    """Small description of an external executable to report.

    Parameters
    ----------
    key:
        Label used in the output report.
    command:
        Executable name or absolute path to resolve.
    version_args:
        Arguments used to query the executable version.
    """

    key: str
    command: str | None = None
    version_args: tuple[str, ...] = ("-version",)

    def resolved_command(self) -> str:
        return self.command or self.key


DEFAULT_EXECUTABLE_SPECS: tuple[ExecutableSpec, ...] = (ExecutableSpec("ffmpeg"),)


def _normalize_executable_specs(
    executables: Iterable[ExecutableSpec | str] | None,
) -> tuple[ExecutableSpec, ...]:
    if executables is None:
        return DEFAULT_EXECUTABLE_SPECS

    normalized: list[ExecutableSpec] = []
    for item in executables:
        if isinstance(item, ExecutableSpec):
            normalized.append(item)
        else:
            normalized.append(ExecutableSpec(str(item)))
    return tuple(normalized)


def collect_executable_summary(
    *,
    executables: Iterable[ExecutableSpec | str] | None = None,
    include_paths: bool = True,
) -> dict[str, str]:
    specs = _normalize_executable_specs(executables)
    summary: dict[str, str] = {}

    for spec in specs:
        key = spec.key
        command = spec.resolved_command()
        summary[key] = _command_version(command, spec.version_args)
        if include_paths:
            summary[f"{key}_path"] = _safe_tail(_which(command))

    return summary


# --------------------------
# Report formatting
# --------------------------


def format_debug_report(
    *,
    sections: Iterable[DebugSection],
    logs_text: str,
) -> str:
    lines: list[str] = []

    for section in sections:
        lines.append(f"## {section.title}")
        if section.items:
            for k, v in section.items.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append("- <no data>")
        lines.append("")

    lines.append("## Recent logs")
    lines.append("```text")
    lines.append(logs_text or "<no captured logs>")
    lines.append("```")

    return "\n".join(lines)


def build_debug_report(
    *,
    recorder: InMemoryDebugRecorder | None,
    libraries: Iterable[LibrarySpec | str] | None = None,
    executables: Iterable[ExecutableSpec | str] | None = None,
    include_module_paths: bool = False,
    include_executable_paths: bool = True,
    log_limit: int = 300,
) -> str:
    logs_text = recorder.render_text(limit=log_limit) if recorder is not None else "<debug recorder unavailable>"

    sections = collect_debug_sections(
        libraries=libraries,
        executables=executables,
        include_module_paths=include_module_paths,
        include_executable_paths=include_executable_paths,
    )

    return format_debug_report(
        sections=sections,
        logs_text=logs_text,
    )


def collect_runtime_summary() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": _safe_tail(sys.executable),
    }


def _section_has_useful_values(items: dict[str, str]) -> bool:
    for value in items.values():
        if value not in {"not-installed", "unknown", "not-found", "unavailable"}:
            return True
    return False


def collect_debug_sections(
    *,
    libraries: Iterable[LibrarySpec | str] | None = None,
    executables: Iterable[ExecutableSpec | str] | None = None,
    include_module_paths: bool = False,
    include_executable_paths: bool = True,
) -> list[DebugSection]:
    sections: list[DebugSection] = []

    # Always include the runtime section first
    sections.append(
        DebugSection(
            title="Runtime",
            items=collect_runtime_summary(),
        )
    )

    # Default grouped report using your built-in constants
    if libraries is None:
        sections.append(
            DebugSection(
                title="DeepLabCut core libraries",
                items=collect_version_summary(
                    libraries=DLC_CORE_LIBS,
                    include_module_paths=include_module_paths,
                ),
            )
        )

        sections.append(
            DebugSection(
                title="GUI libraries",
                items=collect_version_summary(
                    libraries=DLC_GUI_LIBS,
                    include_module_paths=include_module_paths,
                ),
            )
        )

        tf_items = collect_version_summary(
            libraries=DLC_TF_LIBS,
            include_module_paths=include_module_paths,
        )
        if tf_items and _section_has_useful_values(tf_items):
            sections.append(
                DebugSection(
                    title="TensorFlow libraries",
                    items=tf_items,
                )
            )
    else:
        # Custom input
        sections.append(
            DebugSection(
                title="Libraries",
                items=collect_version_summary(
                    libraries=libraries,
                    include_module_paths=include_module_paths,
                ),
            )
        )

    exec_items = collect_executable_summary(
        executables=executables,
        include_paths=include_executable_paths,
    )
    if exec_items and executables is not None:  # report if unavailable
        sections.append(
            DebugSection(
                title="External tools",
                items=exec_items,
            ),
        )

    return sections
