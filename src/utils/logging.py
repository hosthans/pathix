from pathlib import Path
import logging
from datetime import datetime

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.console_colors import ColorFormatter

_DEFAULT_QUIET_LOGGER_NAMES = ("huggingface_hub", "httpx", "requests", "urllib3", "transformers", "timm")

def quiet_external_loggers(names=_DEFAULT_QUIET_LOGGER_NAMES, level: int = logging.WARNING) -> None:
    """Raise third-party loggers' level so they stop emitting INFO/DEBUG noise."""
    for name in names:
        logging.getLogger(name).setLevel(level)


def _resolve_log_path(output_dir: Path, operation: str, force_timestamp: bool) -> Path:
    normalized_operation = "".join(c.lower() if c.isalnum() else "_" for c in operation).strip("_") or "run"
    if force_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return output_dir / f"{normalized_operation}_{timestamp}.log"
    log_file = output_dir / f"{normalized_operation}.log"
    if log_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return output_dir / f"{normalized_operation}_{timestamp}.log"
    return log_file


class _TeeStream:
    """Writes to every wrapped stream (e.g. the real console stream + a log file); used by
    tee_stdout_to_file for entrypoints that print() directly instead of using `logging`.
    """

    def __init__(self, *streams):
        """Store the streams to tee output to."""
        self._streams = streams

    def write(self, data):
        """Write data to every wrapped stream."""
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        """Flush every wrapped stream, ignoring errors."""
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        """Return whether any wrapped stream is a tty."""
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def tee_stdout_to_file(output_dir: Path, operation: str = "run", force_timestamp: bool = True) -> Path:
    """Redirect sys.stdout/sys.stderr so every print() call also lands in a log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = _resolve_log_path(output_dir, operation, force_timestamp)
    handle = open(log_file, "a", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, handle)
    sys.stderr = _TeeStream(sys.__stderr__, handle)
    return log_file


def setup_logging(output_dir: Path, operation: str = "run", level: int = logging.INFO, force_timestamp: bool = False) -> Path:
    """Configure logging to both file and console with consistent naming."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = _resolve_log_path(output_dir, operation, force_timestamp)
    normalized_operation = "".join(c.lower() if c.isalnum() else "_" for c in operation).strip("_") or "run"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            continue
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging configured for operation='{normalized_operation}'. Log file: {log_file}")
    return log_file