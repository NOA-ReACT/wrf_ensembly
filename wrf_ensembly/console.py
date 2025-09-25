import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Logger:
    experiment_path: Path
    command_name: str
    logger: logging.Logger
    log_dir: Path | None = None
    logger = logging.getLogger("rich")

    def __init__(self) -> None:
        logging.basicConfig(
            level="NOTSET",
            format="%(asctime)s: %(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=console, markup=console.is_terminal, rich_tracebacks=True
                )
            ],
        )
        self.logger = logging.getLogger("rich")

    def setup(self, command_name: str, experiment_path: Path):
        self.command_name = command_name
        self.experiment_path = experiment_path

        # Create logger and log directory
        now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.log_dir = experiment_path / "logs" / f"{now}-{command_name}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler to logger
        self.logger.addHandler(logging.FileHandler(self.log_dir / "wrf_ensembly.log"))

    def write_log_file(self, filename: str, contents: str):
        if self.log_dir is None:
            self.logger.warning(
                f"write_log_file(): No log directory set, cannot write log file {filename}"
            )
            return

        log_path = self.log_dir / filename
        with open(log_path, "a") as f:
            f.write(contents)

    def add_log_file(self, source: Path, filename: Optional[str] = None):
        if self.log_dir is None:
            self.logger.warning(
                f"add_log_file(): No log directory set, cannot write log file {source}"
            )
            return

        if filename is None:
            filename = source.name
        shutil.copyfile(source, self.log_dir / filename)

    def info(self, msg: str):
        self.logger.info(msg, stacklevel=2)

    def debug(self, msg: str):
        self.logger.debug(msg, stacklevel=2)

    def error(self, msg: str):
        self.logger.error(msg, stacklevel=2)

    def warning(self, msg: str):
        self.logger.warning(msg, stacklevel=2)


logger = Logger()
