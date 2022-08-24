from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any
from filelock import FileLock, BaseFileLock
import yaml
import time

import pytest

from twister2.device.hardware_map import HardwareMap
from twister2.platform_specification import PlatformSpecification

logger = logging.getLogger(__name__)
logging.getLogger("filelock").setLevel(logging.ERROR)


@dataclass
class TwisterConfig:
    """Store twister configuration to have easy access in test."""
    zephyr_base: str
    output_dir: str = 'twister-out'
    board_root: list = field(default_factory=list)
    build_only: bool = False
    default_platforms: list[str] = field(default_factory=list, repr=False)
    platforms: list[PlatformSpecification] = field(default_factory=list, repr=False)
    # hardware_map_list: list[HardwareMap] = field(default_factory=list, repr=False)
    hardware_map_file: str | None = None
    device_testing: bool = False

    @classmethod
    def create(cls, config: pytest.Config) -> TwisterConfig:
        """Create new instance from pytest.Config."""
        zephyr_base: str = (
            config.getoption('zephyr_base')
            or config.getini('zephyr_base')
            or os.environ.get('ZEPHYR_BASE')
        )
        build_only: bool = config.getoption('--build-only')
        default_platforms: list[str] = config.getoption('--platform')
        board_root: list[str] = config.getoption('--board-root')
        platforms: list[PlatformSpecification] = config._platforms
        output_dir: str = config.getoption('--outdir')
        hardware_map_file: str = config.getoption('--hardware-map')
        device_testing: bool = config.getoption('--device-testing')

        if hardware_map_file:
            cls.clean_hardware_map_availability(hardware_map_file)

        if not default_platforms:
            default_platforms = [
                platform.identifier for platform in platforms
                if platform.testing.default
            ]
        else:
            default_platforms = list(set(default_platforms))  # remove duplicates

        data: dict[str, Any] = dict(
            zephyr_base=zephyr_base,
            build_only=build_only,
            platforms=platforms,
            default_platforms=default_platforms,
            board_root=board_root,
            output_dir=output_dir,
            # hardware_map_list=hardware_map_list,
            hardware_map_file=hardware_map_file,
            device_testing=device_testing,
        )
        return cls(**data)

    def asdict(self) -> dict:
        """Return dictionary which can be serialized as Json."""
        return dict(
            build_only=self.build_only,
            default_platforms=self.default_platforms,
            board_root=self.board_root,
            output_dir=self.output_dir,
        )

    def clean_hardware_map_availability(hardware_map_file: str) -> None:
        """
        Set available statuses of each platform as "True" in case of leaving there "False" value after previous Twister
        call which was accidentally interrupted.
        """
        lock_timeout_duration: float = 600.0  # [s]
        lock: BaseFileLock = FileLock(f"{hardware_map_file}.lock")

        with lock.acquire(lock_timeout_duration):
            with open(hardware_map_file, 'r', encoding='UTF-8') as file:
                data = yaml.safe_load(file)
            hardware_map_list = [HardwareMap(**hardware) for hardware in data]

            for hardware in hardware_map_list:
                if not hardware.available:
                    logger.debug('Set available option value in hardware map to true')
                    hardware.available = True

            with open(hardware_map_file, 'w', encoding='UTF-8') as file:
                hardware_map_list_as_dict = [hardware.asdict() for hardware in hardware_map_list]
                yaml.dump(hardware_map_list_as_dict, file, Dumper=yaml.Dumper, default_flow_style=False)

    def get_hardware_map(self, platform: str) -> HardwareMap | None:
        """
        Return hardware map matching platform and being connected.

        :param platform: platform name
        :return: hardware map or None
        """
        lock_timeout_duration: float = 600.0  # [s]
        lock: BaseFileLock = FileLock(f"{self.hardware_map_file}.lock")

        while True:
            wait_for_platform_flag = False
            with lock.acquire(lock_timeout_duration):
                with open(self.hardware_map_file, 'r', encoding='UTF-8') as file:
                    data = yaml.safe_load(file)
                hardware_map_list = [HardwareMap(**hardware) for hardware in data]

                for hardware in hardware_map_list:
                    if hardware.platform == platform and hardware.connected:
                        if hardware.available:
                            hardware.available = False
                            with open(self.hardware_map_file, 'w', encoding='UTF-8') as file:
                                hardware_map_list_as_dict = [hardware.asdict() for hardware in hardware_map_list]
                                yaml.dump(hardware_map_list_as_dict, file, Dumper=yaml.Dumper, default_flow_style=False)
                            return hardware
                        else:
                            wait_for_platform_flag = True
            if wait_for_platform_flag:
                logger.debug("Waiting for platform %s availability", platform)
                time.sleep(1)
            else:
                return None

    def free_hardware(self, used_hardware: HardwareMap | None) -> None:
        """
        Add mechanism at the beginning of tests to set all devices "available" option to True, to avoid situation when 
        some previous test running leave available option set on False and do not change it to True (in case of
        test running interruption).
        """
        if used_hardware is None:
            return

        lock_timeout_duration: float = 600.0  # [s]
        lock: BaseFileLock = FileLock(f"{self.hardware_map_file}.lock")

        with lock.acquire(lock_timeout_duration):
            with open(self.hardware_map_file, 'r', encoding='UTF-8') as file:
                data = yaml.safe_load(file)
            hardware_map_list = [HardwareMap(**hardware) for hardware in data]

            for hardware in hardware_map_list:
                if hardware.platform == used_hardware.platform \
                        and hardware.id == used_hardware.id \
                        and hardware.product == used_hardware.product:
                    if not hardware.available:
                        hardware.available = True
                        with open(self.hardware_map_file, 'w', encoding='UTF-8') as file:
                            hardware_map_list_as_dict = [hardware.asdict() for hardware in hardware_map_list]
                            yaml.dump(hardware_map_list_as_dict, file, Dumper=yaml.Dumper, default_flow_style=False)
                        break

    def get_platform(self, name: str) -> PlatformSpecification:
        for platform in self.platforms:
            if platform.identifier == name:
                return platform
        raise KeyError(f'There is not platform with identifier: {name}')
