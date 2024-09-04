from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from cuda import cuda, cudart, nvrtc

BlockSpec = tuple[int, int, int]
GridSpec = tuple[int, int, int]


def _cudaGetErrorEnum(error: Any) -> Any:
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result: Any) -> Any:
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0]))
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


_default_device: CudaDevice | None = None
_initialized = False


def do_init(flags: int = 0) -> None:
    global _initialized
    if not _initialized:
        checkCudaErrors(cuda.cuInit(flags))
        _initialized = True


class CudaDevice:
    def __init__(self, device_id: int = 0) -> None:
        do_init()

        self.dev_id = device_id
        self.contexts: list[CudaContext] = []
        self.streams: list[CudaStream] = []

        # Retrieve handle for device 0
        self.nv_device = checkCudaErrors(cuda.cuDeviceGet(device_id))

    def create_context(self) -> CudaContext:
        # Create context
        return CudaContext(self)

    def create_stream(self) -> CudaStream:
        return CudaStream()

    @property
    def name(self) -> str:
        name = cast(bytes, checkCudaErrors(cuda.cuDeviceGetName(512, self.nv_device)))
        return name.decode()

    @property
    def default_context(self) -> CudaContext:
        print("getting default context")
        if len(self.contexts) == 0:
            self.contexts.append(self.create_context())

        return self.contexts[0]

    @property
    def default_stream(self) -> CudaStream:
        if len(self.streams) == 0:
            self.streams.append(self.create_stream())

        return self.streams[0]

    @property
    def compute_capability(self) -> tuple[int, int]:
        # Derive target architecture for device 0
        major = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.nv_device
            )
        )
        minor = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.nv_device
            )
        )
        return (major, minor)

    @property
    def driver_version(self) -> tuple[int, int]:
        version_num = checkCudaErrors(cuda.cuDriverGetVersion())
        major = version_num // 1000
        minor = (version_num - (major * 1000)) // 10
        return (major, minor)

    @staticmethod
    def default() -> CudaDevice:
        global _default_device
        if _default_device is None:
            _default_device = CudaDevice(0)
        return _default_device

    @staticmethod
    def count() -> int:
        do_init()

        return cast(int, checkCudaErrors(cuda.cuDeviceGetCount()))


class CudaContext:
    def __init__(self, dev: CudaDevice) -> None:
        print("creating context")
        self.nv_context = checkCudaErrors(cuda.cuCtxCreate(0, dev.nv_device))

    def __del__(self) -> None:
        checkCudaErrors(cuda.cuCtxDestroy(self.nv_context))


class CudaStream:
    def __init__(self, flags: int = cuda.CUstream_flags.CU_STREAM_DEFAULT) -> None:
        self.nv_stream = checkCudaErrors(cuda.cuStreamCreate(flags))

    def __del__(self) -> None:
        self.synchronize()
        checkCudaErrors(cuda.cuStreamDestroy(self.nv_stream))

    def synchronize(self) -> None:
        checkCudaErrors(cuda.cuStreamSynchronize(self.nv_stream))

    # is_done
    # wait_for_event


class CudaEvent:
    pass
    # record
    # synchronize


class CudaMemory:
    # managed
    # pagelocked
    pass
    # malloc
    # to_device
    # from_device
    # free
    # as_buffer


class GraphNode(ABC):
    @abstractmethod
    def __nv_mknode__(self, graph: Any) -> None: ...


class KernelNode(GraphNode):
    def __init__(
        self,
        fn: CudaFunction,
        *,
        block: BlockSpec = (1, 1, 1),
        grid: GridSpec = (1, 1, 1),
        device: CudaDevice | None = None,
    ) -> None:
        self.block = block
        self.grid = grid
        self.device = device
        self.fn = fn

    def __nv_mknode__(self, nv_graph: Any) -> None:
        kernelNodeParams = cuda.CUDA_KERNEL_NODE_PARAMS()
        kernelNodeParams.func = self.fn.kernel  # type: ignore
        kernelNodeParams.gridDimX = self.grid[0]
        kernelNodeParams.gridDimY = self.grid[1]
        kernelNodeParams.gridDimZ = self.grid[2]
        kernelNodeParams.blockDimX = self.block[0]
        kernelNodeParams.blockDimY = self.block[1]
        kernelNodeParams.blockDimZ = self.block[2]
        kernelNodeParams.sharedMemBytes = 0
        # kernelNodeParams.kernelParams = kernelArgs
        kernelNodeParams.kernelParams = 0
        checkCudaErrors(cuda.cuGraphAddKernelNode(nv_graph, None, 0, kernelNodeParams))


class CudaGraph:
    # https://github.com/NVIDIA/cuda-python/blob/main/examples/3_CUDA_Features/simpleCudaGraphs_test.py
    def __init__(self, *, stream: CudaStream | None = None) -> None:
        if stream is None:
            dev = CudaDevice.default()
            stream = dev.default_stream
        self.stream = stream
        self.nv_graph = checkCudaErrors(cuda.cuGraphCreate(0))

    def run(self) -> None:
        self.graphExec = checkCudaErrors(cudart.cudaGraphInstantiate(self.nv_graph, 0))
        checkCudaErrors(cudart.cudaGraphLaunch(self.graphExec, self.stream.nv_stream))

    def add_node(self, n: GraphNode) -> None:
        self.nodes.append(n)
        n.__nv_mknode__(self)

    # cuStreamBeginCaptureToGraph
    # instantiate()
    # upload()
    # launch()

    def add_kernel_node(self, fn: CudaFunction) -> None:
        kernelNodeParams = cuda.CUDA_KERNEL_NODE_PARAMS()
        self.fn = fn
        kernelNodeParams.func = fn.kernel  # type: ignore
        kernelNodeParams.gridDimX = 1
        kernelNodeParams.gridDimY = kernelNodeParams.gridDimZ = 1
        kernelNodeParams.blockDimX = 1
        kernelNodeParams.blockDimY = kernelNodeParams.blockDimZ = 1
        kernelNodeParams.sharedMemBytes = 0
        # kernelNodeParams.kernelParams = kernelArgs
        kernelNodeParams.kernelParams = 0
        checkCudaErrors(cuda.cuGraphAddKernelNode(self.nv_graph, None, 0, kernelNodeParams))

    # add_memcpy_node()
    # add_memset_node()
    # add_host_node()
    # add_child_graph_node()
    # add_empty_node()
    # add_event_record_node()
    # add_event_wait_node()
    # add_external_semaphore_node()
    # add_external_semaphore_wait_node()
    # add_batch_memop_node()
    # add_mem_alloc_node()
    # add_mem_free_node()
    # mem_trim()
    # clone()
    @property
    def nodes(self) -> list[object]:
        nodes, numNodes = checkCudaErrors(cudart.cudaGraphGetNodes(self.nv_graph))
        return cast(list[object], nodes)

    # root_nodes[]
    # edges[]
    # to_dot()
    # to_networkx()


NvDataType = type[ctypes.c_uint] | type[ctypes.c_void_p]


class CudaData:
    def __init__(self, data: int, datatype: NvDataType | None = None) -> None:
        self.data = data
        if datatype is None:
            datatype = ctypes.c_uint
        self.type = datatype


class CudaFunction:
    def __init__(self, src: CudaSource, name: str) -> None:
        self.src = src
        self.name = name
        self.kernel = checkCudaErrors(cuda.cuModuleGetFunction(src.module, name.encode()))


class CudaSource:
    # TODO: include paths, compiler flags
    def __init__(
        self, code: str, *, no_extern: bool = False, progname: str = "<unspecified>"
    ) -> None:
        device = CudaDevice.default()
        context = device.default_context  # cuModuleLoadData requires a context
        stream = device.default_stream
        self.progname = progname
        if not no_extern:
            self.code = 'extern "C" {\n' + code + "\n}\n"
        print(f"CODE:\n-------\n{self.code}\n-------\n")

        # Create program
        self.prog = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(self.code.encode(), self.progname.encode(), 0, [], [])
        )

        # Compile code
        # Compile program
        # arch_arg = bytes(f"--gpu-architecture=compute_{major}{minor}", "ascii")
        # opts = [b"--fmad=false", arch_arg]
        # ret = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        compile_result = nvrtc.nvrtcCompileProgram(self.prog, 0, [])
        log_sz = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(self.prog))
        buf = b" " * log_sz
        checkCudaErrors(nvrtc.nvrtcGetProgramLog(self.prog, buf))
        self.compile_log = buf.decode()
        if log_sz > 0:
            print(f"Compilation results:\b{self.compile_log}")
        else:
            print("Compilation complete, no warnings.")
        checkCudaErrors(compile_result)

        # Get PTX from compilation
        self.ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(self.prog))
        self.ptx = b" " * self.ptx_size
        checkCudaErrors(nvrtc.nvrtcGetPTX(self.prog, self.ptx))

    def get_function(
        self,
        name: str,
        *,
        device: CudaDevice | None = None,
    ) -> CudaFunction:
        if device is None:
            device = CudaDevice.default()
        context = device.default_context  # cuModuleLoadData requires a context
        stream = device.default_stream

        # Load PTX as module data and retrieve function
        self.ptx = np.char.array(self.ptx)
        self.module = checkCudaErrors(cuda.cuModuleLoadData(self.ptx.ctypes.data))

        return CudaFunction(self, name)

    def call(
        self,
        name: str,
        args: CudaData | list[CudaData] | None = None,
        *,
        block: BlockSpec = (1, 1, 1),
        grid: GridSpec = (1, 1, 1),
        device: CudaDevice | None = None,
    ) -> None:
        if device is None:
            device = CudaDevice.default()
        context = device.default_context  # cuModuleLoadData requires a context
        stream = device.default_stream

        print("Calling function:", name)
        fn = self.get_function(name, device=device)

        NvArgs = tuple[
            tuple[Any, ...],  # list of data
            tuple[Any, ...],  # list of data types
        ]
        if args is None:
            nv_args: NvArgs | int = 0
        else:
            if isinstance(args, CudaData):
                args = [args]

            nv_data_args = tuple(arg.data for arg in args)
            nv_type_args = tuple(arg.type for arg in args)
            nv_args = (nv_data_args, nv_type_args)

        checkCudaErrors(
            cuda.cuLaunchKernel(
                fn.kernel,
                grid[0],  # grid x dim
                grid[1],  # grid y dim
                grid[2],  # grid z dim
                block[0],  # block x dim
                block[1],  # block y dim
                block[2],  # block z dim
                0,  # dynamic shared memory
                stream.nv_stream,  # stream
                #    args.ctypes.data,  # kernel arguments
                nv_args,  # kernel arguments
                0,  # extra (ignore)
            )
        )

    def __del__(self) -> None:
        checkCudaErrors(cuda.cuModuleUnload(self.module))


class CudaSourceFile(CudaSource):
    def __init__(self, filename: str) -> None:
        with open(filename) as f:
            code = f.read()
        super().__init__(code=code, progname=filename)
