"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow.core.framework.graph_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class TensorTracerReport(google.protobuf.message.Message):
    """Tensor Tracer Report proto gives information about the trace including:
    - TensorTracerConfig: version, device, num replicas, trace mode.
    - Graphdef, e.g., list of operations, tensors
    - TracedTensorDef:
       * Name of the tensor
       * Tracepoint name if provided.
       * Index of the tensor in the compact cache if traced.
       * Explanation for why the tensor is traced or not.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class TensordefEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> global___TensorTracerReport.TracedTensorDef: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[global___TensorTracerReport.TracedTensorDef] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    class TensorTracerConfig(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        VERSION_FIELD_NUMBER: builtins.int
        DEVICE_FIELD_NUMBER: builtins.int
        TRACE_MODE_FIELD_NUMBER: builtins.int
        NUM_CORES_FIELD_NUMBER: builtins.int
        NUM_HOSTS_FIELD_NUMBER: builtins.int
        SUBMODE_FIELD_NUMBER: builtins.int
        NUM_CORES_PER_HOST_FIELD_NUMBER: builtins.int
        INCLUDED_CORES_FIELD_NUMBER: builtins.int
        SIGNATURES_FIELD_NUMBER: builtins.int
        version: typing.Text
        """Tensor tracer version, e.g. hostcall, outside compilation."""

        device: typing.Text
        """Traced device, CPU, TPU..."""

        trace_mode: typing.Text
        """Trace mode, norm, summary, full-trace."""

        num_cores: builtins.int
        """Number of cores, e.g. TPU cores, in the system."""

        num_hosts: builtins.int
        """Number of hosts, e.g. compute nodes in the system."""

        submode: typing.Text
        """Keep submode as string for backward compatibility."""

        num_cores_per_host: builtins.int
        """Keep num cores per host for backward compatibility."""

        @property
        def included_cores(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
            """Id of the included cores, if a subset of cores are traced."""
            pass
        @property
        def signatures(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
            """The names of the signatures corresponding to the cache indices."""
            pass
        def __init__(self,
            *,
            version: typing.Text = ...,
            device: typing.Text = ...,
            trace_mode: typing.Text = ...,
            num_cores: builtins.int = ...,
            num_hosts: builtins.int = ...,
            submode: typing.Text = ...,
            num_cores_per_host: builtins.int = ...,
            included_cores: typing.Optional[typing.Iterable[builtins.int]] = ...,
            signatures: typing.Optional[typing.Iterable[typing.Text]] = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["device",b"device","included_cores",b"included_cores","num_cores",b"num_cores","num_cores_per_host",b"num_cores_per_host","num_hosts",b"num_hosts","signatures",b"signatures","submode",b"submode","trace_mode",b"trace_mode","version",b"version"]) -> None: ...

    class TracedTensorDef(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        NAME_FIELD_NUMBER: builtins.int
        CACHE_INDEX_FIELD_NUMBER: builtins.int
        TRACE_POINT_NAME_FIELD_NUMBER: builtins.int
        IS_TRACED_FIELD_NUMBER: builtins.int
        EXPLANATION_FIELD_NUMBER: builtins.int
        name: typing.Text
        """Name of the tensor as appears in tf graph."""

        cache_index: builtins.int
        """Cache index of the tensor. This may be different than topological index."""

        trace_point_name: typing.Text
        """If trace points are provided, corresponding tracepoint name of the
        tensor. Trace points are placed on the edges (tensors) in the tensorflow
        graph, and they force tensor tracer to trace the corresponding tensor.
        Tracepoints can be added using the programatic interface
        tensor_tracer.tensor_tracepoint(tensor, trace_point_name) function.
        This will add a trace point with the given trace_point_name for the given
        tensor. If a trace_point is provided for the tensor,
        trace_point name will be used for the rest of the analysis instead of
        tensor names. One can use trace_point_name's to compare two models with
        arbitrary tensor names by providing the same trace point name for the
        tensors that are comparable.
        """

        is_traced: builtins.bool
        """Whether the tensor is traced or not."""

        explanation: typing.Text
        """Detailed explanation why the tensor is traced or not."""

        def __init__(self,
            *,
            name: typing.Text = ...,
            cache_index: builtins.int = ...,
            trace_point_name: typing.Text = ...,
            is_traced: builtins.bool = ...,
            explanation: typing.Text = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["cache_index",b"cache_index","explanation",b"explanation","is_traced",b"is_traced","name",b"name","trace_point_name",b"trace_point_name"]) -> None: ...

    CONFIG_FIELD_NUMBER: builtins.int
    GRAPHDEF_FIELD_NUMBER: builtins.int
    TENSORDEF_FIELD_NUMBER: builtins.int
    FINGERPRINT_FIELD_NUMBER: builtins.int
    @property
    def config(self) -> global___TensorTracerReport.TensorTracerConfig: ...
    @property
    def graphdef(self) -> tensorflow.core.framework.graph_pb2.GraphDef:
        """Tensorflow graph."""
        pass
    @property
    def tensordef(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___TensorTracerReport.TracedTensorDef]:
        """A map from tensor name to its TracedTensorDef."""
        pass
    fingerprint: typing.Text
    """The fingerprint of the TensorTracerReport (fingerprint calculation excludes
    this field and graphdef).
    """

    def __init__(self,
        *,
        config: typing.Optional[global___TensorTracerReport.TensorTracerConfig] = ...,
        graphdef: typing.Optional[tensorflow.core.framework.graph_pb2.GraphDef] = ...,
        tensordef: typing.Optional[typing.Mapping[typing.Text, global___TensorTracerReport.TracedTensorDef]] = ...,
        fingerprint: typing.Text = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["config",b"config","graphdef",b"graphdef"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["config",b"config","fingerprint",b"fingerprint","graphdef",b"graphdef","tensordef",b"tensordef"]) -> None: ...
global___TensorTracerReport = TensorTracerReport