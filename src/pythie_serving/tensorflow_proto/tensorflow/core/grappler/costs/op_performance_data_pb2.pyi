"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow.core.framework.attr_value_pb2
import tensorflow.core.framework.tensor_pb2
import tensorflow.core.framework.tensor_shape_pb2
import tensorflow.core.framework.types_pb2
import tensorflow.core.protobuf.device_properties_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class SessionInfo(google.protobuf.message.Message):
    """Description of the session when an op is run."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    INTRA_OP_PARALLELISM_FIELD_NUMBER: builtins.int
    intra_op_parallelism: builtins.int
    def __init__(self,
        *,
        intra_op_parallelism: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["intra_op_parallelism",b"intra_op_parallelism"]) -> None: ...
global___SessionInfo = SessionInfo

class OpInfo(google.protobuf.message.Message):
    """Description of an operation as well as the parameters expected to impact its
    performance.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class AttrEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> tensorflow.core.framework.attr_value_pb2.AttrValue: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[tensorflow.core.framework.attr_value_pb2.AttrValue] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    class TensorProperties(google.protobuf.message.Message):
        """Input data types, shapes and values if known."""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        DTYPE_FIELD_NUMBER: builtins.int
        SHAPE_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        dtype: tensorflow.core.framework.types_pb2.DataType.ValueType
        @property
        def shape(self) -> tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto: ...
        @property
        def value(self) -> tensorflow.core.framework.tensor_pb2.TensorProto: ...
        def __init__(self,
            *,
            dtype: tensorflow.core.framework.types_pb2.DataType.ValueType = ...,
            shape: typing.Optional[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto] = ...,
            value: typing.Optional[tensorflow.core.framework.tensor_pb2.TensorProto] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["shape",b"shape","value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["dtype",b"dtype","shape",b"shape","value",b"value"]) -> None: ...

    OP_FIELD_NUMBER: builtins.int
    ATTR_FIELD_NUMBER: builtins.int
    INPUTS_FIELD_NUMBER: builtins.int
    OUTPUTS_FIELD_NUMBER: builtins.int
    DEVICE_FIELD_NUMBER: builtins.int
    SESSION_INFO_FIELD_NUMBER: builtins.int
    op: typing.Text
    """The operation name.  There may be custom parameters in attrs."""

    @property
    def attr(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, tensorflow.core.framework.attr_value_pb2.AttrValue]:
        """Custom parameters impacting the behavior of the op."""
        pass
    @property
    def inputs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___OpInfo.TensorProperties]: ...
    @property
    def outputs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___OpInfo.TensorProperties]:
        """Optional description of the op outputs"""
        pass
    @property
    def device(self) -> tensorflow.core.protobuf.device_properties_pb2.DeviceProperties:
        """Device on which the operation is run."""
        pass
    @property
    def session_info(self) -> global___SessionInfo:
        """Information about the session configs."""
        pass
    def __init__(self,
        *,
        op: typing.Text = ...,
        attr: typing.Optional[typing.Mapping[typing.Text, tensorflow.core.framework.attr_value_pb2.AttrValue]] = ...,
        inputs: typing.Optional[typing.Iterable[global___OpInfo.TensorProperties]] = ...,
        outputs: typing.Optional[typing.Iterable[global___OpInfo.TensorProperties]] = ...,
        device: typing.Optional[tensorflow.core.protobuf.device_properties_pb2.DeviceProperties] = ...,
        session_info: typing.Optional[global___SessionInfo] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["device",b"device","session_info",b"session_info"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["attr",b"attr","device",b"device","inputs",b"inputs","op",b"op","outputs",b"outputs","session_info",b"session_info"]) -> None: ...
global___OpInfo = OpInfo

class NormalDistribution(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    MU_FIELD_NUMBER: builtins.int
    SIGMA_FIELD_NUMBER: builtins.int
    mu: builtins.float
    sigma: builtins.float
    def __init__(self,
        *,
        mu: builtins.float = ...,
        sigma: builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["mu",b"mu","sigma",b"sigma"]) -> None: ...
global___NormalDistribution = NormalDistribution

class LogNormalDistribution(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    MU_FIELD_NUMBER: builtins.int
    SIGMA_FIELD_NUMBER: builtins.int
    mu: builtins.float
    sigma: builtins.float
    def __init__(self,
        *,
        mu: builtins.float = ...,
        sigma: builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["mu",b"mu","sigma",b"sigma"]) -> None: ...
global___LogNormalDistribution = LogNormalDistribution

class OpPerformance(google.protobuf.message.Message):
    """Performance data for tensorflow operations"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class OpMemory(google.protobuf.message.Message):
        """Memory usage data for a tensorflow operation."""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        OUTPUT_MEMORY_FIELD_NUMBER: builtins.int
        TEMP_MEMORY_FIELD_NUMBER: builtins.int
        PERSISTENT_MEMORY_FIELD_NUMBER: builtins.int
        DEVICE_TEMP_MEMORY_FIELD_NUMBER: builtins.int
        DEVICE_PERSISTENT_MEMORY_FIELD_NUMBER: builtins.int
        @property
        def output_memory(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
            """The output information may have memory usage and output shapes."""
            pass
        temp_memory: builtins.int
        """Temp and persistent memory allocated by this node."""

        persistent_memory: builtins.int
        device_temp_memory: builtins.int
        device_persistent_memory: builtins.int
        def __init__(self,
            *,
            output_memory: typing.Optional[typing.Iterable[builtins.int]] = ...,
            temp_memory: builtins.int = ...,
            persistent_memory: builtins.int = ...,
            device_temp_memory: builtins.int = ...,
            device_persistent_memory: builtins.int = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["device_persistent_memory",b"device_persistent_memory","device_temp_memory",b"device_temp_memory","output_memory",b"output_memory","persistent_memory",b"persistent_memory","temp_memory",b"temp_memory"]) -> None: ...

    OP_FIELD_NUMBER: builtins.int
    SESSION_INFO_FIELD_NUMBER: builtins.int
    NODE_FIELD_NUMBER: builtins.int
    TEMPORARY_MEMORY_SIZE_FIELD_NUMBER: builtins.int
    COMPUTE_COST_FIELD_NUMBER: builtins.int
    COMPUTE_TIME_FIELD_NUMBER: builtins.int
    MEMORY_TIME_FIELD_NUMBER: builtins.int
    COMPUTE_EFFICIENCY_FIELD_NUMBER: builtins.int
    MEMORY_EFFICIENCY_FIELD_NUMBER: builtins.int
    EXECUTION_TIME_NORMAL_FIELD_NUMBER: builtins.int
    EXECUTION_TIME_LOG_NORMAL_FIELD_NUMBER: builtins.int
    OP_MEMORY_FIELD_NUMBER: builtins.int
    @property
    def op(self) -> global___OpInfo:
        """The op"""
        pass
    @property
    def session_info(self) -> global___SessionInfo:
        """Information about the session configs."""
        pass
    node: typing.Text
    """The node name (optional). Makes it easier to associate the performance data
    with a specific graph node.
    """

    temporary_memory_size: builtins.int
    """Temporary memory used by this node (in bytes)."""

    compute_cost: builtins.int
    """Time it takes to run the op (in nanoseconds)."""

    compute_time: builtins.int
    """Analytical compute cost (in nanoseconds)."""

    memory_time: builtins.int
    """Analytical memory access cost (in nanoseconds)."""

    compute_efficiency: builtins.float
    """Percentage of theoretical compute performance."""

    memory_efficiency: builtins.float
    """Percentage of theoretical memory performance."""

    @property
    def execution_time_normal(self) -> global___NormalDistribution: ...
    @property
    def execution_time_log_normal(self) -> global___LogNormalDistribution: ...
    @property
    def op_memory(self) -> global___OpPerformance.OpMemory: ...
    def __init__(self,
        *,
        op: typing.Optional[global___OpInfo] = ...,
        session_info: typing.Optional[global___SessionInfo] = ...,
        node: typing.Text = ...,
        temporary_memory_size: builtins.int = ...,
        compute_cost: builtins.int = ...,
        compute_time: builtins.int = ...,
        memory_time: builtins.int = ...,
        compute_efficiency: builtins.float = ...,
        memory_efficiency: builtins.float = ...,
        execution_time_normal: typing.Optional[global___NormalDistribution] = ...,
        execution_time_log_normal: typing.Optional[global___LogNormalDistribution] = ...,
        op_memory: typing.Optional[global___OpPerformance.OpMemory] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["execution_time",b"execution_time","execution_time_log_normal",b"execution_time_log_normal","execution_time_normal",b"execution_time_normal","op",b"op","op_memory",b"op_memory","session_info",b"session_info"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["compute_cost",b"compute_cost","compute_efficiency",b"compute_efficiency","compute_time",b"compute_time","execution_time",b"execution_time","execution_time_log_normal",b"execution_time_log_normal","execution_time_normal",b"execution_time_normal","memory_efficiency",b"memory_efficiency","memory_time",b"memory_time","node",b"node","op",b"op","op_memory",b"op_memory","session_info",b"session_info","temporary_memory_size",b"temporary_memory_size"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["execution_time",b"execution_time"]) -> typing.Optional[typing_extensions.Literal["execution_time_normal","execution_time_log_normal"]]: ...
global___OpPerformance = OpPerformance

class OpPerformanceList(google.protobuf.message.Message):
    """A collection of OpPerformance data points."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    OP_PERFORMANCE_FIELD_NUMBER: builtins.int
    @property
    def op_performance(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___OpPerformance]: ...
    def __init__(self,
        *,
        op_performance: typing.Optional[typing.Iterable[global___OpPerformance]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["op_performance",b"op_performance"]) -> None: ...
global___OpPerformanceList = OpPerformanceList