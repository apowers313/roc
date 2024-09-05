from __future__ import annotations

import ctypes
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Any, NewType, cast

import numpy as np
from cuda import cuda, cudart, nvrtc


def _cudaGetErrorEnum(error: Any) -> Any:
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result: tuple[Any, ...]) -> Any:
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
        self.nv_device: NvDevice = checkCudaErrors(cuda.cuDeviceGet(device_id))

    def create_context(self) -> CudaContext:
        # Create context
        return CudaContext(self)

    def create_stream(self) -> CudaStream:
        return CudaStream()

    @property
    def name(self) -> str:
        name: bytes = checkCudaErrors(cuda.cuDeviceGetName(512, self.nv_device))
        return name.decode()

    @property
    def default_context(self) -> CudaContext:
        # TODO: cuDevicePrimaryCtxRetain?
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
        self.nv_context: NvContext = checkCudaErrors(cuda.cuCtxCreate(0, dev.nv_device))

    def __del__(self) -> None:
        checkCudaErrors(cuda.cuCtxDestroy(self.nv_context))


class CudaStream:
    def __init__(self, flags: int = cuda.CUstream_flags.CU_STREAM_DEFAULT) -> None:
        self.nv_stream: NvStream = checkCudaErrors(cuda.cuStreamCreate(flags))

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
    def __init__(self, size: int, ctx: CudaContext | None = None) -> None:
        if ctx is None:
            device = CudaDevice.default()
            ctx = device.default_context

        self.size = size
        self.nv_memory: NvMemory = checkCudaErrors(cudart.cudaMalloc(size))
        # self.nv_memory: NvMemory = checkCudaErrors(cuda.cuMemAlloc(size))

    # def __del__(self) -> None:
    #     checkCudaErrors(cuda.cuMemFree(self.nv_memory))

    @staticmethod
    def from_np(arr: numpy.ndarray, *, stream: CudaStream | None = None) -> CudaMemory:
        if stream is None:
            dev = CudaDevice.default()
            stream = dev.default_stream

        num_bytes = len(arr) * arr.itemsize
        mem = CudaMemory(num_bytes)
        # print("mem.nv_memory", mem.nv_memory)
        # print("arr.ctypes.data", arr.ctypes.data)
        # print("num_bytes", num_bytes)
        # print("stream", stream)
        checkCudaErrors(
            cuda.cuMemcpyHtoDAsync(mem.nv_memory, arr.ctypes.data, num_bytes, stream.nv_stream)
        )

        return mem

    # cuda.cuMemcpy
    # cuda.cuMemcpyHtoD
    # cuda.cuMemcpyDtoH

    # managed
    # pagelocked
    pass
    # malloc
    # to_device
    # from_device
    # free
    # as_buffer


class GraphNode(metaclass=ABCMeta):
    @abstractmethod
    def __nv_mknode__(self, graph: Any) -> None:
        pass


class KernelNode(GraphNode):
    def __init__(
        self,
        fn: CudaFunction,
        args: KernelArgs = None,
        *,
        block: BlockSpec = (1, 1, 1),
        grid: GridSpec = (1, 1, 1),
    ) -> None:
        self.block = block
        self.grid = grid
        self.fn = fn
        self.args = args
        self.nv_args = make_args(args)

        nv_kernel_node_params = cuda.CUDA_KERNEL_NODE_PARAMS()
        nv_kernel_node_params.func = self.fn.nv_kernel
        nv_kernel_node_params.gridDimX = self.grid[0]
        nv_kernel_node_params.gridDimY = self.grid[1]
        nv_kernel_node_params.gridDimZ = self.grid[2]
        nv_kernel_node_params.blockDimX = self.block[0]
        nv_kernel_node_params.blockDimY = self.block[1]
        nv_kernel_node_params.blockDimZ = self.block[2]
        nv_kernel_node_params.sharedMemBytes = 0
        nv_kernel_node_params.kernelParams = self.nv_args
        self.nv_kernel_node_params = nv_kernel_node_params

        self.nv_kernel_node: NvKernelNode | None = None

    def __nv_mknode__(self, graph: CudaGraph) -> None:
        self.nv_kernel_node = checkCudaErrors(
            cuda.cuGraphAddKernelNode(graph.nv_graph, None, 0, self.nv_kernel_node_params)
        )


class MallocNode(GraphNode):
    def __init__(self, size: int) -> None:
        self.size = size
        self.nv_memory: NvMallocNode | None = None

        nv_memalloc_params = cudart.cudaMemAllocNodeParams()
        nv_memalloc_params.bytesize = size
        # nv_memalloc_params.poolProps
        # nv_memalloc_params.accessDescs
        # nv_memalloc_params.accessDescCount
        # nv_memalloc_params.dptr
        # nv_memalloc_params.getPtr()
        self.nv_memalloc_params = nv_memalloc_params

    def __nv_mknode__(self, graph: CudaGraph) -> None:
        self.nv_memory = checkCudaErrors(
            cudart.cudaGraphAddMemAllocNode(graph.nv_graph, None, 0, self.nv_memalloc_params)
        )


class CopyDirection(Enum):
    device_to_host = 1
    host_to_device = 2


class MemcpyNode(GraphNode):
    def __init__(self, src: CudaMemory, dst: CudaMemory, size: int, direction: str) -> None:
        self.src = src
        self.dst = dst
        self.size = size
        self.direction = CopyDirection[direction]
        self.nv_src = self.src.nv_memory
        self.nv_dst = self.dst.nv_memory
        self.nv_memcpy_node: NvMemcpyNode | None = None

        nv_memcpy_params = cudart.cudaMemcpy3DParms()
        nv_memcpy_params.srcArray = None
        nv_memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
        nv_memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(
            self.nv_src, np.dtype(np.float64).itemsize, 1, 1
        )
        nv_memcpy_params.dstArray = None
        nv_memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
        nv_memcpy_params.dstPtr = cudart.make_cudaPitchedPtr(
            self.nv_dst, np.dtype(np.float64).itemsize, 1, 1
        )
        nv_memcpy_params.extent = cudart.make_cudaExtent(np.dtype(np.float64).itemsize, 1, 1)
        if self.direction == CopyDirection.device_to_host:
            nv_memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        else:
            nv_memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        self.nv_memcpy_params = nv_memcpy_params

    def __nv_mknode__(self, graph: CudaGraph) -> None:
        self.nv_memcpy_node = checkCudaErrors(
            cudart.cudaGraphAddMemcpyNode(graph.nv_graph, None, 0, self.nv_memcpy_params)
        )


class MemsetNode(GraphNode):
    def __init__(self, mem: CudaMemory, value: int, size: int) -> None:
        self.size = size
        self.value = value
        self.nv_memset_node: NvMemsetNode | None = None

        nv_memset_params = cudart.cudaMemsetParams()
        nv_memset_params.dst = mem.nv_memory
        nv_memset_params.value = self.value
        # nv_memset_params.elementSize = np.dtype(np.float32).itemsize
        nv_memset_params.elementSize = np.dtype(np.uint8).itemsize
        nv_memset_params.width = self.size
        nv_memset_params.height = 1
        self.nv_memset_params = nv_memset_params

    def __nv_mknode__(self, graph: CudaGraph) -> None:
        self.nv_memset_node = checkCudaErrors(
            cudart.cudaGraphAddMemsetNode(graph.nv_graph, None, 0, self.nv_memset_params)
        )


class CudaGraph:
    # https://github.com/NVIDIA/cuda-python/blob/main/examples/3_CUDA_Features/simpleCudaGraphs_test.py
    def __init__(self, *, stream: CudaStream | None = None) -> None:
        if stream is None:
            dev = CudaDevice.default()
            stream = dev.default_stream
        self.stream = stream
        self.nodes: list[GraphNode] = []
        self.nv_graph: NvGraph = checkCudaErrors(cuda.cuGraphCreate(0))

    def run(self) -> None:
        self.nv_graph_exec: NvGraphExec = checkCudaErrors(
            cudart.cudaGraphInstantiate(self.nv_graph, 0)
        )

        checkCudaErrors(cudart.cudaGraphLaunch(self.nv_graph_exec, self.stream.nv_stream))

    def add_node(self, n: GraphNode) -> None:
        self.nodes.append(n)
        n.__nv_mknode__(self)

    # cuStreamBeginCaptureToGraph
    # instantiate()
    # upload()
    # launch()

    # def add_kernel_node(self, fn: CudaFunction) -> None:
    #     kernelNodeParams = cuda.CUDA_KERNEL_NODE_PARAMS()
    #     self.fn = fn
    #     kernelNodeParams.func = fn.kernel  # type: ignore
    #     kernelNodeParams.gridDimX = 1
    #     kernelNodeParams.gridDimY = kernelNodeParams.gridDimZ = 1
    #     kernelNodeParams.blockDimX = 1
    #     kernelNodeParams.blockDimY = kernelNodeParams.blockDimZ = 1
    #     kernelNodeParams.sharedMemBytes = 0
    #     # kernelNodeParams.kernelParams = kernelArgs
    #     kernelNodeParams.kernelParams = 0
    #     checkCudaErrors(cuda.cuGraphAddKernelNode(self.nv_graph, None, 0, kernelNodeParams))

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
    def nv_nodes(self) -> list[NvGraphNode]:
        nodes: list[NvGraphNode]
        numNodes: int
        nodes, numNodes = checkCudaErrors(cudart.cudaGraphGetNodes(self.nv_graph))
        return nodes

    # root_nodes[]
    # edges[]
    # to_dot()
    # to_networkx()


NvDataType = type[ctypes.c_uint] | type[ctypes.c_void_p]


class CudaData:
    def __init__(self, data: int | CudaMemory, datatype: NvDataType | None = None) -> None:
        if isinstance(data, int):
            self.data: int | NvMemory = data
        if isinstance(data, CudaMemory):
            self.data = data.nv_memory
            datatype = ctypes.c_void_p

        if datatype is None:
            datatype = ctypes.c_uint
        self.type = datatype


class CudaFunction:
    def __init__(self, src: CudaSource, name: str) -> None:
        self.src = src
        self.name = name
        self.nv_kernel: NvKernel = checkCudaErrors(
            cuda.cuModuleGetFunction(src.nv_module, name.encode())
        )


def make_args(args: KernelArgs) -> NvKernelArgs:
    if args is None:
        nv_args: NvKernelArgs | int = 0
    else:
        if isinstance(args, CudaData):
            args = [args]

        nv_data_args = tuple(arg.data for arg in args)
        nv_type_args = tuple(arg.type for arg in args)
        nv_args = (nv_data_args, nv_type_args)
    return nv_args


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
        self.nv_prog: NvProgram = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(self.code.encode(), self.progname.encode(), 0, [], [])
        )

        # Compile code
        # Compile program
        # arch_arg = bytes(f"--gpu-architecture=compute_{major}{minor}", "ascii")
        # opts = [b"--fmad=false", arch_arg]
        # ret = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        compile_result = nvrtc.nvrtcCompileProgram(self.nv_prog, 0, [])
        log_sz = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(self.nv_prog))
        buf = b" " * log_sz
        checkCudaErrors(nvrtc.nvrtcGetProgramLog(self.nv_prog, buf))
        self.compile_log = buf.decode()
        if log_sz > 0:
            print(f"Compilation results:\b{self.compile_log}")
        else:
            print("Compilation complete, no warnings.")
        checkCudaErrors(compile_result)

        # Get PTX from compilation
        self.nv_ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(self.nv_prog))
        self.ptx = b" " * self.nv_ptx_size
        checkCudaErrors(nvrtc.nvrtcGetPTX(self.nv_prog, self.ptx))

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
        self.nv_module = checkCudaErrors(cuda.cuModuleLoadData(self.ptx.ctypes.data))

        return CudaFunction(self, name)

    def call(
        self,
        name: str,
        args: KernelArgs = None,
        *,
        block: BlockSpec = (1, 1, 1),
        grid: GridSpec = (1, 1, 1),
        stream: CudaStream | None = None,
    ) -> None:
        if stream is None:
            device = CudaDevice.default()
            stream = device.default_stream

        print("Calling function:", name)
        fn = self.get_function(name, device=device)

        nv_args = make_args(args)

        print("nv_args", nv_args)

        checkCudaErrors(
            cuda.cuLaunchKernel(
                fn.nv_kernel,
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
        checkCudaErrors(cuda.cuModuleUnload(self.nv_module))


class CudaSourceFile(CudaSource):
    def __init__(self, filename: str) -> None:
        with open(filename) as f:
            code = f.read()
        super().__init__(code=code, progname=filename)


# internal types
KernelArgs = CudaData | list[CudaData] | None
BlockSpec = tuple[int, int, int]
GridSpec = tuple[int, int, int]

# types for CUDA Python
NvDevice = NewType("NvDevice", object)  # cuda.CUdevice
NvContext = NewType("NvContext", object)  # cuda.CUcontext
NvStream = NewType("NvStream", object)  # cuda.CUstream
NvGraphExec = NewType("NvGraphExec", object)  # cuda.CUgraphExec
NvGraph = NewType("NvGraph", object)  # cuda.CUgraph
NvGraphNode = NewType("NvGraphNode", object)  # cuda.CUgraphNode
NvKernel = NewType("NvKernel", object)  # cuda.CUkernel
NvProgram = NewType("NvProgram", object)  # nvrtc.nvrtcGetProgram
NvMemory = NewType("NvMemory", object)  # cuda.CUdeviceptr
NvKernelNode = NewType("NvKernelNode", object)
NvMallocNode = NewType("NvMallocNode", object)
NvMemcpyNode = NewType("NvMemcpyNode", object)
NvMemsetNode = NewType("NvMemsetNode", object)
NvKernelArgs = (
    int  # for None args
    | tuple[
        tuple[Any, ...],  # list of data
        tuple[Any, ...],  # list of data types
    ]
)
