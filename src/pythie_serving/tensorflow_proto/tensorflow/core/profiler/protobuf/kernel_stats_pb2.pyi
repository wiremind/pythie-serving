"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class KernelReport(google.protobuf.message.Message):
    """Next ID: 15"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NAME_FIELD_NUMBER: builtins.int
    REGISTERS_PER_THREAD_FIELD_NUMBER: builtins.int
    STATIC_SHMEM_BYTES_FIELD_NUMBER: builtins.int
    DYNAMIC_SHMEM_BYTES_FIELD_NUMBER: builtins.int
    BLOCK_DIM_FIELD_NUMBER: builtins.int
    GRID_DIM_FIELD_NUMBER: builtins.int
    TOTAL_DURATION_NS_FIELD_NUMBER: builtins.int
    MIN_DURATION_NS_FIELD_NUMBER: builtins.int
    MAX_DURATION_NS_FIELD_NUMBER: builtins.int
    IS_KERNEL_USING_TENSOR_CORE_FIELD_NUMBER: builtins.int
    IS_OP_TENSOR_CORE_ELIGIBLE_FIELD_NUMBER: builtins.int
    OP_NAME_FIELD_NUMBER: builtins.int
    OCCURRENCES_FIELD_NUMBER: builtins.int
    OCCUPANCY_PCT_FIELD_NUMBER: builtins.int
    name: typing.Text
    """Name of the kernel."""

    registers_per_thread: builtins.int
    """Registers per thread."""

    static_shmem_bytes: builtins.int
    """Static shared memory in bytes."""

    dynamic_shmem_bytes: builtins.int
    """Dynamic shared memory in bytes."""

    @property
    def block_dim(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Block dimensions."""
        pass
    @property
    def grid_dim(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Grid dimensions."""
        pass
    total_duration_ns: builtins.int
    """Total duration of this kernel."""

    min_duration_ns: builtins.int
    """Min duration of kernel in nanoseconds."""

    max_duration_ns: builtins.int
    """Max duration of kernel in nanoseconds."""

    is_kernel_using_tensor_core: builtins.bool
    """Kernel utilizes TensorCore instructions."""

    is_op_tensor_core_eligible: builtins.bool
    """Operation is eligible to use TensorCores."""

    op_name: typing.Text
    """TF operation name."""

    occurrences: builtins.int
    """Number of occurrences."""

    occupancy_pct: builtins.float
    """Occupancy percentage."""

    def __init__(self,
        *,
        name: typing.Text = ...,
        registers_per_thread: builtins.int = ...,
        static_shmem_bytes: builtins.int = ...,
        dynamic_shmem_bytes: builtins.int = ...,
        block_dim: typing.Optional[typing.Iterable[builtins.int]] = ...,
        grid_dim: typing.Optional[typing.Iterable[builtins.int]] = ...,
        total_duration_ns: builtins.int = ...,
        min_duration_ns: builtins.int = ...,
        max_duration_ns: builtins.int = ...,
        is_kernel_using_tensor_core: builtins.bool = ...,
        is_op_tensor_core_eligible: builtins.bool = ...,
        op_name: typing.Text = ...,
        occurrences: builtins.int = ...,
        occupancy_pct: builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["block_dim",b"block_dim","dynamic_shmem_bytes",b"dynamic_shmem_bytes","grid_dim",b"grid_dim","is_kernel_using_tensor_core",b"is_kernel_using_tensor_core","is_op_tensor_core_eligible",b"is_op_tensor_core_eligible","max_duration_ns",b"max_duration_ns","min_duration_ns",b"min_duration_ns","name",b"name","occupancy_pct",b"occupancy_pct","occurrences",b"occurrences","op_name",b"op_name","registers_per_thread",b"registers_per_thread","static_shmem_bytes",b"static_shmem_bytes","total_duration_ns",b"total_duration_ns"]) -> None: ...
global___KernelReport = KernelReport

class KernelStatsDb(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    REPORTS_FIELD_NUMBER: builtins.int
    @property
    def reports(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___KernelReport]:
        """A list of kernels aggregated by name."""
        pass
    def __init__(self,
        *,
        reports: typing.Optional[typing.Iterable[global___KernelReport]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["reports",b"reports"]) -> None: ...
global___KernelStatsDb = KernelStatsDb