import dataclasses
import os
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import click
import psutil

from roc.attention import Attention, SaliencyMap, VisionAttentionData
from roc.component import Component
from roc.event import Event
from roc.graphdb import Edge, Node

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


class NodeCacheState(State[float]):
    def __init__(self) -> None:
        super().__init__("node-cache", display_name="Node Cache")
        self.val = 0

    def get(self) -> float:
        c = Node.get_cache()
        return c.currsize / c.maxsize

    def __str__(self) -> str:
        c = Node.get_cache()
        return f"Node Cache: {c.currsize} / {c.maxsize} ({self.get():1.1f}%)"


class EdgeCacheState(State[float]):
    def __init__(self) -> None:
        super().__init__("edge-cache", display_name="Edge Cache")
        self.val = 0

    def get(self) -> float:
        c = Edge.get_cache()
        return c.currsize / c.maxsize

    def __str__(self) -> str:
        c = Edge.get_cache()
        return f"Edge Cache: {c.currsize} / {c.maxsize} ({self.get():1.1f}%)"


class CurrentScreenState(State[str]):
    def __init__(self) -> None:
        super().__init__("curr-screen", display_name="Current Screen")

    def set(self, screen: str) -> None:
        self.val = screen

    def __str__(self) -> str:
        if self.val is not None:
            return f"Current Screen:\n-------------\n{self.val}\n-------------"
        else:
            return "Current Screen: None"


class CurrentSaliencyMapState(State[SaliencyMap]):
    def __init__(self) -> None:
        super().__init__("curr-saliency", display_name="Current Saliency Map")

    def set(self, sal: SaliencyMap) -> None:
        self.val = sal

    def __str__(self) -> str:
        if self.val is not None:
            s = f"Current Saliency Map:\n{str(self.val)}\n"
            s += "\tFeatures:\n"
            features = self.val.feature_report()
            for feat_name in features:
                s += f"\t\t{feat_name}: {features[feat_name]}\n"
            return s
        else:
            return "Current Saliency Map: None"


class CurrentAttentionState(State[VisionAttentionData]):
    def __init__(self) -> None:
        super().__init__("curr-saliency", display_name="Current Saliency Map")

    def set(self, att: VisionAttentionData) -> None:
        self.val = att

    def __str__(self) -> str:
        if self.val is not None:
            s = f"Current Attention:\n{str(self.val)}\n"
            s += f"\tCurrent Focus: {self.val.focus_points}\n"
            return s
        else:
            return "Current Attention: None"


class ComponentsState(State[list[str]]):
    def __init__(self) -> None:
        super().__init__("components", display_name="Components")
        self.val = []

    def get(self) -> list[str]:
        self.val = Component.get_loaded_components()
        return self.val

    def __str__(self) -> str:
        component_str = "\t" + "\n\t".join(self.get())
        return f"{Component.get_component_count()} components loaded:\n{component_str}"


@dataclass
class StateList:
    memory: ProcessMemoryState = ProcessMemoryState()
    sysmem: AvailableMemoryState = AvailableMemoryState()
    loop: LoopState = LoopState()
    cpuload: CpuLoadState = CpuLoadState()
    diskio: DiskIoState = DiskIoState()
    node_cache: NodeCacheState = NodeCacheState()
    edge_cache: EdgeCacheState = EdgeCacheState()
    screen: CurrentScreenState = CurrentScreenState()
    salency: CurrentSaliencyMapState = CurrentSaliencyMapState()
    attention: CurrentAttentionState = CurrentAttentionState()
    components: ComponentsState = ComponentsState()


states = StateList()
all_states = [field.name for field in dataclasses.fields(StateList)]


class StateComponent(Component):
    pass


_state_init_done = False


def init_state() -> None:
    global _state_init_done
    if _state_init_done:
        return

    att_conn = Attention.bus.connect(StateComponent())

    def att_evt_handler(e: Event[VisionAttentionData]) -> None:
        assert isinstance(e.data, VisionAttentionData)
        states.salency.set(deepcopy(e.data.saliency_map))
        states.attention.set(deepcopy(e.data))

    att_conn.listen(att_evt_handler, filter=lambda e: isinstance(e.data, VisionAttentionData))
    _state_init_done = True


def print_state() -> None:
    init_state()

    def header(s: str) -> None:
        print(f"\n=== {s.upper()} ===")  # noqa: T201

    header("System Health")
    print(states.cpuload)  # noqa: T201
    print(states.diskio)  # noqa: T201
    print(states.memory)  # noqa: T201
    print(states.sysmem)  # noqa: T201

    header("Environment")
    print(states.loop)  # noqa: T201
    print(states.screen)  # noqa: T201
    # TODO: blstats

    header("Graph DB")
    print(states.node_cache)  # noqa: T201
    print(states.edge_cache)  # noqa: T201

    header("Agent")
    print(states.components)  # noqa: T201
    print(states.salency)  # noqa: T201
    print(states.attention)  # noqa: T201


@click.command()
@click.argument(
    "var",
    nargs=-1,
    type=click.Choice(all_states, case_sensitive=False),
)
def state_cli(var: list[str]) -> None:
    if var is None or len(var) < 1:
        # if no state is specified, print a selection of the most interesting states
        print_state()
        return

    for v in var:
        s = getattr(states, v)
        print(str(s))  # noqa: T201
        pass
