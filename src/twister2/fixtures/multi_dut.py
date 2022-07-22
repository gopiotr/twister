from __future__ import annotations

import logging
from typing import Type, Generator
import multiprocessing as mp
from pathlib import Path

import pytest

from twister2.device.device_abstract import DeviceAbstract
from twister2.device.factory import DeviceFactory
from twister2.twister_config import TwisterConfig
from twister2.exceptions import TwisterConfigurationException
from twister2.yaml_test_specification import YamlTestSpecification

logger = logging.getLogger(__name__)


@pytest.fixture(scope='function')
def multi_dut(request: pytest.FixtureRequest) -> DeviceAbstract:
    """Return list of device instances."""
    twister_config: TwisterConfig = request.config.twister_config
    yaml_spec: YamlTestSpecification = request.function.spec

    devices: list[DeviceAbstract] = []

    bsim_config: dict = yaml_spec.bsim_config
    if bsim_config:
        devices = [device for device in _initialize_bsim_devices(twister_config, yaml_spec)]
    else:
        err_msg = 'No specification for multi device'
        logger.error(err_msg)
        raise TwisterConfigurationException(err_msg)

    build_dir: Path = twister_config.bsim_bin_path

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    number_processes = len(devices)
    pool = mp.Pool(processes=number_processes)

    if not twister_config.build_only:
        for device in devices:
            pool.apply_async(device.flash, [build_dir])

    pool.close()
    pool.join()

    yield devices
    if not twister_config.build_only:
        for device in devices:
            device.disconnect()


def _initialize_bsim_devices(twister_config: TwisterConfig, yaml_spec: YamlTestSpecification) -> Generator[DeviceAbstract, None, None]:
    bsim_config: dict = yaml_spec.bsim_config
    simulation_id: str = yaml_spec.name
    simulation_id = simulation_id.replace(".", "_").replace("[", "_").replace("]", "_")

    for idx, bsim_device in enumerate(bsim_config['devices']):
        exe_name = bsim_device["exe_name"]
        exe_args = [
            f"-s={simulation_id}",
            f"-d={idx}",
            f"-testid={bsim_device['testid']}"
        ]
        device_class: Type[DeviceAbstract] = DeviceFactory.get_device('bsim')
        device = device_class(
            twister_config=twister_config,
            exe_name=exe_name,
            exe_args=exe_args,
            hardware_map=None
        )
        yield device

    # PHY
    bsim_phy = bsim_config["physical_layer"]
    exe_name = bsim_phy["exe_name"]
    exe_args = [
        f"-s={simulation_id}",
        f"-D={len(bsim_config['devices'])}",
        f"-sim_length={bsim_phy['sim_length']}"
    ]
    device_class: Type[DeviceAbstract] = DeviceFactory.get_device('bsim')
    device = device_class(
        twister_config=twister_config,
        exe_name=exe_name,
        exe_args=exe_args,
        hardware_map=None
    )
    yield device
