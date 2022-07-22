from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Generator

from twister2.device.native_simulator_adapter import NativeSimulatorAdapter
from twister2.twister_config import TwisterConfig
from twister2.device.hardware_map import HardwareMap

logger = logging.getLogger(__name__)


class BsimAdapter(NativeSimulatorAdapter):
    def __init__(self, twister_config: TwisterConfig, exe_name: str, exe_args: list, hardware_map: HardwareMap | None = None, **kwargs):
        super().__init__(twister_config, hardware_map, **kwargs)
        self.exe_name: str = exe_name
        self.exe_args: list = exe_args
        short_name: str = exe_name.split("/")[-1] + exe_args[-1] + ".log"
        out_dir_path = Path(twister_config.zephyr_base) / "bsim-out"
        out_dir_path.mkdir(parents=True, exist_ok=True)
        self.out_file_path = out_dir_path / short_name

    def get_command(self, build_dir: Path | str) -> list:
        exe_path: Path = build_dir / self.exe_name
        return [str(exe_path)] + self.exe_args

    def _collect_process_output(self, process: subprocess.Popen) -> threading.Thread:
        """Create Thread which saves a process output to a file."""
        def _read():
            with process.stdout:
                with open(self.out_file_path, "w") as f:
                    for line in iter(process.stdout.readline, b''):
                        f.write(line.decode())

        return threading.Thread(target=_read, daemon=True)

    @property
    def out(self) -> Generator[str, None, None]:
        """Return output from serial."""
        with open(self.out_file_path, "r") as f:
            for line in f.readlines():
                yield line
