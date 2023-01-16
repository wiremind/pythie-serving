"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
This proto describes the types of hardware profiled by the TensorFlow
profiler.
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _HardwareType:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _HardwareTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_HardwareType.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    UNKNOWN_HARDWARE: _HardwareType.ValueType  # 0
    """Unknown hardware."""
    CPU_ONLY: _HardwareType.ValueType  # 1
    """CPU only without any hardware accelerator."""
    GPU: _HardwareType.ValueType  # 2
    """GPU."""
    TPU: _HardwareType.ValueType  # 3
    """TPU."""

class HardwareType(_HardwareType, metaclass=_HardwareTypeEnumTypeWrapper):
    """Types of hardware profiled."""

UNKNOWN_HARDWARE: HardwareType.ValueType  # 0
"""Unknown hardware."""
CPU_ONLY: HardwareType.ValueType  # 1
"""CPU only without any hardware accelerator."""
GPU: HardwareType.ValueType  # 2
"""GPU."""
TPU: HardwareType.ValueType  # 3
"""TPU."""
global___HardwareType = HardwareType

@typing_extensions.final
class GPUComputeCapability(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MAJOR_FIELD_NUMBER: builtins.int
    MINOR_FIELD_NUMBER: builtins.int
    major: builtins.int
    minor: builtins.int
    def __init__(
        self,
        *,
        major: builtins.int = ...,
        minor: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["major", b"major", "minor", b"minor"]) -> None: ...

global___GPUComputeCapability = GPUComputeCapability

@typing_extensions.final
class DeviceCapabilities(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CLOCK_RATE_IN_GHZ_FIELD_NUMBER: builtins.int
    NUM_CORES_FIELD_NUMBER: builtins.int
    MEMORY_SIZE_IN_BYTES_FIELD_NUMBER: builtins.int
    MEMORY_BANDWIDTH_FIELD_NUMBER: builtins.int
    COMPUTE_CAPABILITY_FIELD_NUMBER: builtins.int
    DEVICE_VENDOR_FIELD_NUMBER: builtins.int
    clock_rate_in_ghz: builtins.float
    num_cores: builtins.int
    memory_size_in_bytes: builtins.int
    memory_bandwidth: builtins.int
    """Bytes/s."""
    @property
    def compute_capability(self) -> global___GPUComputeCapability: ...
    device_vendor: builtins.str
    def __init__(
        self,
        *,
        clock_rate_in_ghz: builtins.float = ...,
        num_cores: builtins.int = ...,
        memory_size_in_bytes: builtins.int = ...,
        memory_bandwidth: builtins.int = ...,
        compute_capability: global___GPUComputeCapability | None = ...,
        device_vendor: builtins.str = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["compute_capability", b"compute_capability"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["clock_rate_in_ghz", b"clock_rate_in_ghz", "compute_capability", b"compute_capability", "device_vendor", b"device_vendor", "memory_bandwidth", b"memory_bandwidth", "memory_size_in_bytes", b"memory_size_in_bytes", "num_cores", b"num_cores"]) -> None: ...

global___DeviceCapabilities = DeviceCapabilities
