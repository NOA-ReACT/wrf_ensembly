import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from rich.console import Console
from rich.logging import RichHandler

console = Console()


@dataclass
class LoggerConfig:
    experiment_path: Path
    command_name: str


def get_logger(cfg: LoggerConfig | None) -> Tuple[logging.Logger, Path | None]:
    """
    Get a logger with a rich (console) handler and, optionally, a file handler.

    If you know the experiment directory, pass the `cfg` argument to create an directory
    inside `experiment/logs` that corresponds to the current date/time/command name.
    The logs will be kept automatically. You can also store any other files
    (i.e., stdout, rsl.*, ...) in the same directory.

    Returns:
        The logger and the log directory (if `cfg` is not None).
    """
    handlers = [
        RichHandler(console=console, markup=console.is_terminal, rich_tracebacks=True)
    ]
    log_dir = None
    if cfg is not None:
        now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_dir = cfg.experiment_path / "logs" / f"{now}-{cfg.command_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "wrf_ensembly.log"))

    logging.basicConfig(
        level="NOTSET",
        format="%(asctime)s: %(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    return logging.getLogger("rich"), log_dir
