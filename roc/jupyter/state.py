import dataclasses
import os
import time
from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import click
import psutil

from .utils import bytes2human

StateType = TypeVar("StateType")


class State(ABC, Generic[StateType]):
    def __init__(self, name: str, display_name: str | None = None) -> None:
        self.name = name
        self.display_name = display_name or name
        self.val: StateType | None = None

    def __str__(self) -> str:
        return f"{self.display_name}: {self.val}"

    def get(self) -> StateType:
        if self.val is None:
            raise Exception("Trying to get state value before it is set")

        return self.val

    def set(self, v: StateType) -> None:
        self.val = v


class SystemCpuState(State[int]):
    def __init__(self) -> None:
        super().__init__("cpu", display_name="CPU Usage")
        self.val = self.get()

    def get(self) -> int:
        psutil.cpu_times()
        return 1


class ProcessMemoryState(State[int]):
    def __init__(self) -> None:
        super().__init__("memory")
        self.val = self.get()

    def __str__(self) -> str:
        mem = self.get()
        return f"Process Memory Usage: {bytes2human(mem)}"

    def get(self) -> int:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return cast(int, mem_info.rss)


class AvailableMemoryState(State[int]):
    def __init__(self) -> None:
        super().__init__("sysmem")
        self.val = self.get()

    def __str__(self) -> str:
        vm = self.get()
        return f"Available System Memory: {bytes2human(vm)}"

    def get(self) -> int:
        vm = psutil.virtual_memory()
        return cast(int, vm.available)


class CpuLoadState(State[list[float]]):
    def __init__(self) -> None:
        super().__init__("cpuload")
        self.val = self.get()

    def __str__(self) -> str:
        load = self.get()
        return f"CPU Load: {load[0]:1.1f}% / {load[1]:1.1f}% / {load[2]:1.1f}% (1m / 5m / 15m)"

    def get(self) -> list[float]:
        return [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]


class DiskIoState(State[dict[str, float]]):
    def __init__(self) -> None:
        super().__init__("diskio")
        self.last_time = time.time_ns()
        ioc = psutil.disk_io_counters()
        self.last_read_io = ioc.read_count
        self.last_write_io = ioc.write_count
        self.last_read_bytes = ioc.read_bytes
        self.last_write_bytes = ioc.write_bytes

    def __str__(self) -> str:
        disk_io = self.get()
        read_io = disk_io["read_io"]
        write_io = disk_io["write_io"]
        read_bytes = int(disk_io["read_bytes"])
        write_bytes = int(disk_io["write_bytes"])
        return f"Disk Read I/O: {read_io:1.1f}/s ({bytes2human(read_bytes)}/s), Write I/O {write_io:1.1f}/s ({bytes2human(write_bytes)}/s)"

    def get(self) -> dict[str, float]:
        ioc = psutil.disk_io_counters()
        now = time.time_ns()
        delta_sec = (now - self.last_time) / 10e8
        read_io_per_sec = (ioc.read_count - self.last_read_io) / delta_sec
        write_io_per_sec = (ioc.write_count - self.last_write_io) / delta_sec
        read_bytes_per_sec = (ioc.read_bytes - self.last_read_bytes) / delta_sec
        write_bytes_per_sec = (ioc.write_bytes - self.last_write_bytes) / delta_sec
        self.last_read_io = ioc.read_count
        self.last_write_io = ioc.write_count
        self.last_read_bytes = ioc.read_bytes
        self.last_write_bytes = ioc.write_bytes
        self.last_time = now

        return {
            "read_io": read_io_per_sec,
            "write_io": write_io_per_sec,
            "read_bytes": read_bytes_per_sec,
            "write_bytes": write_bytes_per_sec,
        }


class LoopState(State[int]):
    def __init__(self) -> None:
        super().__init__("loop", display_name="Loop Number")
        self.val = 0

    def incr(self) -> None:
        self.val = self.get() + 1


@dataclass
class StateList:
    memory: ProcessMemoryState = ProcessMemoryState()
    sysmem: AvailableMemoryState = AvailableMemoryState()
    loop: LoopState = LoopState()
    cpuload: CpuLoadState = CpuLoadState()
    diskio: DiskIoState = DiskIoState()
    # node cache size / hits / misses
    # edge cache size / hits / misses
    # current screen
    # current saliency map


states = StateList()


@click.command()
@click.argument(
    "var",
    type=click.Choice(
        [field.name for field in dataclasses.fields(StateList)], case_sensitive=False
    ),
)
def state_cli(var: list[str]) -> None:
    for v in var:
        pass
