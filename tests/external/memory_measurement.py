from __future__ import annotations

import argparse
import csv
import logging
import matplotlib.pyplot as plt
import os
import psutil
import shlex
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import NamedTuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
logger.addHandler(ch)

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR_NAME: Path = CURRENT_DIR / "measurements"
TIMESTAMP_FORMAT: str = "%H:%M:%S.%f"
TIMESTAMP_INFO_FILE_BASE_NAME: str = "timestamp_info.csv"


class ExperimentInfo(NamedTuple):
    name: str
    command: str
    cwd: str


class ProcessInfo(NamedTuple):
    pid: int
    name: str
    cmdline: str


class TimestampInfo:
    def __init__(self, timestamp: str) -> None:
        self.timestamp: str = timestamp
        self.pid_rss_mbytes: dict[int, float] = {}

    def get_rss_mbytes_sum(self) -> float:
        return sum(self.pid_rss_mbytes.values())


def run_experiment(experiment_name: str, command: list[str], cwd: str, out_dir: Path) -> None:
    main_process: subprocess.Popen = _run_process(command, cwd)
    measurements, processes_info = _collect_data(main_process)
    _save_measurements(out_dir, measurements, processes_info, experiment_name)


def _run_process(command: list[str], cwd: str) -> subprocess.Popen:
    logger.info("Run command: %s", shlex.join(command))
    env = os.environ.copy()
    process = subprocess.Popen(command, env=env, text=True, cwd=cwd,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    return process


def _collect_data(main_process: subprocess.Popen) -> tuple[list[TimestampInfo], dict[int, ProcessInfo]]:
    measurements: list[TimestampInfo] = []
    processes_info: dict[int, ProcessInfo] = {}  # key is a process PID

    main_ps_pid = main_process.pid
    main_ps_tracker: psutil.Process = psutil.Process(main_ps_pid)

    # add info about main process
    main_ps_info = ProcessInfo(main_ps_pid, main_ps_tracker.name(), main_ps_tracker.cmdline())
    processes_info[main_ps_pid] = main_ps_info

    while main_process.poll() is None:
        timestamp: str = datetime.now().strftime(TIMESTAMP_FORMAT)
        measurement = TimestampInfo(timestamp)

        # analyze main process
        try:
            measurement.pid_rss_mbytes[main_ps_pid] = _to_mb(main_ps_tracker.memory_info().rss)
        except psutil.NoSuchProcess:
            pass

        # analyze child processes
        for child_ps_tracker in main_ps_tracker.children(recursive=True):
            try:
                child_ps_pid = child_ps_tracker.pid
                if child_ps_pid not in processes_info:
                    child_ps_info = ProcessInfo(child_ps_pid, child_ps_tracker.name(), child_ps_tracker.cmdline())
                    processes_info[child_ps_pid] = child_ps_info

                measurement.pid_rss_mbytes[child_ps_pid] = _to_mb(child_ps_tracker.memory_info().rss)
            except psutil.NoSuchProcess:
                pass

        measurements.append(measurement)
        time.sleep(0.1)

    # just to be sure that main process was finished:
    main_process.wait()

    msg = f"Main process finished with return code: {main_process.returncode}"
    if main_process.returncode == 0:
        logger.info(msg)
    else:
        logger.error(msg)
        for line in main_process.stdout.readlines():
            logger.debug(line)

    return measurements, processes_info


def _to_mb(bytes: int) -> float:
    """
    Convert from bytest to Mbytes
    """
    return bytes / (1024 * 1024)


def _save_measurements(
        out_dir: Path,
        measurements: list[TimestampInfo],
        processes_info: dict[int, ProcessInfo],
        experiment_name: str
) -> None:
    out_dir = out_dir / experiment_name
    os.makedirs(out_dir, exist_ok=True)
    _save_to_files(measurements, processes_info, out_dir)
    _draw_plot(measurements, out_dir, experiment_name)


def _save_to_files(
        measurements: list[TimestampInfo],
        processes_info: dict[int, ProcessInfo],
        out_dir: Path
) -> None:
    logger.info("Save measurements to files")

    with open(out_dir / "processes_info.txt", "w") as file:
        for process_info in processes_info.values():
            file.write(f"{process_info.pid}\n{process_info.name}\n{process_info.cmdline}\n")
            for measurement in measurements:
                if process_info.pid in measurement.pid_rss_mbytes:
                    file.write(f"{measurement.timestamp}: {str(measurement.pid_rss_mbytes[process_info.pid])}\n")
            file.write("\n")

    with open(out_dir / "timestamp_info_full.txt", "w") as file:
        for measurement in measurements:
            file.write(f"{measurement.timestamp}, {measurement.get_rss_mbytes_sum()}\n")
            for pid, rss_mbytes in measurement.pid_rss_mbytes.items():
                process_info = processes_info[pid]
                file.write(f"{process_info.pid} {rss_mbytes} {process_info.name} {process_info.cmdline}\n")
            file.write("\n")

    with open(out_dir / TIMESTAMP_INFO_FILE_BASE_NAME, "w") as file:
        writer = csv.writer(file)
        header = ["timestamp", "rss_mbytes"]
        writer.writerow(header)
        for measurement in measurements:
            writer.writerow([measurement.timestamp, measurement.get_rss_mbytes_sum()])


def _draw_plot(
        measurements: list[TimestampInfo],
        out_dir: Path,
        experiment_name: str
) -> None:
    logger.info("Draw plot")

    timestamps = [measurement.timestamp for measurement in measurements]
    timestamps = _normalize_timestamps(timestamps)
    rss_mbytes = [measurement.get_rss_mbytes_sum() for measurement in measurements]
    plt.clf()
    plt.plot(timestamps, rss_mbytes)
    plt.title(f"{experiment_name}")
    plt.xlabel("time [s]")
    plt.ylabel("RSS [Mbytes]")
    plt.grid()
    plt.savefig(out_dir / "measurement.png")
    # plt.show()
    plt.clf()


def _normalize_timestamps(timestamps_raw: list[str]) -> list[float]:
    timestamps_datetime = [datetime.strptime(t, TIMESTAMP_FORMAT) for t in timestamps_raw]
    timestamps_datetime_sec = [t.timestamp() for t in timestamps_datetime]
    first_timestamps_datetime_sec = timestamps_datetime_sec[0]
    timestamps_sec = [t - first_timestamps_datetime_sec for t in timestamps_datetime_sec]
    return timestamps_sec


def draw_comparison_plot(out_dir: Path) -> None:
    logger.info("Prepare comparison")

    plt.clf()
    experiment_dirs = [dir for dir in out_dir.iterdir() if dir.is_dir()]
    for experiment_dir in experiment_dirs:
        experiment_name = experiment_dir.name
        file_name = experiment_dir / TIMESTAMP_INFO_FILE_BASE_NAME
        timestamps = []
        rss_mbytes = []
        with open(file_name) as file:
            reader = csv.DictReader(file)
            for line in reader:
                timestamps.append(line["timestamp"])
                rss_mbytes.append(float(line["rss_mbytes"]))
        timestamps = _normalize_timestamps(timestamps)
        plt.plot(timestamps, rss_mbytes, label=experiment_name)

    plt.title("comparison")
    plt.xlabel("time [s]")
    plt.ylabel("RSS [Mbytes]")
    plt.legend()
    plt.grid()
    plt.savefig(out_dir / "comparison.png")
    plt.show()
    plt.clf()


def get_zephyr_base_path() -> str:
    zephyr_base_path = os.getenv("ZEPHYR_BASE")
    if zephyr_base_path:
        return zephyr_base_path
    else:
        logger.error("Please set ZEPHYR_BASE environment variable")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-od", "--out-dir", default=DEFAULT_OUT_DIR_NAME, type=Path,
                        help="Output directory where all data will be saved")

    parser.add_argument("-co", "--compare-only", action="store_true",
                        help="Skip running measurements, compare only historical data")

    return parser.parse_args()


def main():
    args = parse_args()
    compare_only: bool = args.compare_only
    out_dir: Path = args.out_dir

    zephyr_base_path = get_zephyr_base_path()

    # v1_basic_command = ["./scripts/twister", "-vv"]
    # v2_basic_command = ["pytest", "-vvs", "--log-level=INFO", "-o", "log_cli=true"]
    v1_basic_command = ["./scripts/twister"]
    v2_basic_command = ["pytest"]

    experiments = [
        # # collect only
        # ExperimentInfo("v1_np_ker_com_co_j4", v1_basic_command + ["-p", "native_posix", "-T", "tests/kernel/common", "--dry-run"], zephyr_base_path),
        # ExperimentInfo("v2_np_ker_com_co_n4", v2_basic_command + ["--platform=native_posix", "tests/kernel/common", "-n", "4", "--collect-only"], zephyr_base_path),

        # # build only
        # ExperimentInfo("v1_np_ker_com_bo_j4", v1_basic_command + ["-p", "native_posix", "-T", "tests/kernel/common", "--build-only"], zephyr_base_path),
        # ExperimentInfo("v2_np_ker_com_bo_n4", v2_basic_command + ["--platform=native_posix", "tests/kernel/common", "-n", "4", "--build-only"], zephyr_base_path),

        # execute test
        ExperimentInfo("v1_np_ker_com", v1_basic_command + ["-p", "native_posix", "-T", "tests/kernel/common"], zephyr_base_path),
        ExperimentInfo("v2_np_ker_com_n_auto", v2_basic_command + ["--platform=native_posix", "tests/kernel/common", "-n", "auto"], zephyr_base_path),
    ]

    # out_dir = CURRENT_DIR / "collect_only"
    # out_dir = CURRENT_DIR / "build_only"
    # out_dir = CURRENT_DIR / "execute"

    os.makedirs(out_dir, exist_ok=True)

    if not compare_only:
        for experiment in experiments:
            run_experiment(*experiment, out_dir)

    draw_comparison_plot(out_dir)


if __name__ == "__main__":
    main()
