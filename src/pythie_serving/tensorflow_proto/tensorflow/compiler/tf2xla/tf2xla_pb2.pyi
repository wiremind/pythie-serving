"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow.core.framework.tensor_shape_pb2
import tensorflow.core.framework.types_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class TensorId(google.protobuf.message.Message):
    """TensorId identifies a tensor in a TensorFlow graph, by specifying the output
    index of a particular node in the graph.  If the output of the named node
    feeds into other node(s), this corresponds to one or more edges.  Otherwise
    it doesn't correspond to any existing edges at all, e.g. for output nodes.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NODE_NAME_FIELD_NUMBER: builtins.int
    OUTPUT_INDEX_FIELD_NUMBER: builtins.int
    node_name: typing.Text
    output_index: builtins.int
    def __init__(self,
        *,
        node_name: typing.Text = ...,
        output_index: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["node_name",b"node_name","output_index",b"output_index"]) -> None: ...
global___TensorId = TensorId

class Feed(google.protobuf.message.Message):
    """Feed represents a single feed tensor in the graph, which corresponds to an
    input argument for the generated computation.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ID_FIELD_NUMBER: builtins.int
    SHAPE_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    @property
    def id(self) -> global___TensorId: ...
    @property
    def shape(self) -> tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto: ...
    name: typing.Text
    """Optional name for generated code."""

    type: tensorflow.core.framework.types_pb2.DataType.ValueType
    """Optional data type. This is not normally required, as the graph itself
    contains this information. However, if the node being fed is an op that is
    not linked into the binary, then the type cannot be inferred from the node;
    in this case, the type should be set here.
    """

    def __init__(self,
        *,
        id: typing.Optional[global___TensorId] = ...,
        shape: typing.Optional[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto] = ...,
        name: typing.Text = ...,
        type: tensorflow.core.framework.types_pb2.DataType.ValueType = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["id",b"id","shape",b"shape"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["id",b"id","name",b"name","shape",b"shape","type",b"type"]) -> None: ...
global___Feed = Feed

class Fetch(google.protobuf.message.Message):
    """Fetch represents a single fetch tensor in the graph, which corresponds to an
    output argument for the generated computation.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ID_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    SHAPE_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    @property
    def id(self) -> global___TensorId: ...
    name: typing.Text
    """Optional name for generated code."""

    @property
    def shape(self) -> tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto:
        """Optional shape and data type. If specified, may be used for validation."""
        pass
    type: tensorflow.core.framework.types_pb2.DataType.ValueType
    def __init__(self,
        *,
        id: typing.Optional[global___TensorId] = ...,
        name: typing.Text = ...,
        shape: typing.Optional[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto] = ...,
        type: tensorflow.core.framework.types_pb2.DataType.ValueType = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["id",b"id","shape",b"shape"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["id",b"id","name",b"name","shape",b"shape","type",b"type"]) -> None: ...
global___Fetch = Fetch

class Variable(google.protobuf.message.Message):
    """Variable represents a resource variable with the given name, shape and type."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NODE_NAME_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    SHAPE_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    READONLY_FIELD_NUMBER: builtins.int
    node_name: typing.Text
    name: typing.Text
    """Optional name for generated code. If empty, node_name will be used."""

    @property
    def shape(self) -> tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto: ...
    type: tensorflow.core.framework.types_pb2.DataType.ValueType
    readonly: builtins.bool
    """Flag for variables that are never assigned. Assignments to a read-only
    variable or unassigned variables that are not read-only are invalid.
    """

    def __init__(self,
        *,
        node_name: typing.Text = ...,
        name: typing.Text = ...,
        shape: typing.Optional[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto] = ...,
        type: tensorflow.core.framework.types_pb2.DataType.ValueType = ...,
        readonly: builtins.bool = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["shape",b"shape"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["name",b"name","node_name",b"node_name","readonly",b"readonly","shape",b"shape","type",b"type"]) -> None: ...
global___Variable = Variable

class ConversionOptions(google.protobuf.message.Message):
    """Options used during the conversion and compilation process."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    CUSTOM_FAKE_QUANT_OP_CALLS_FIELD_NUMBER: builtins.int
    custom_fake_quant_op_calls: builtins.bool
    """When true tf.fake_quant_* ops will be emitted as custom calls to a
    'fake_quant_with_min_max_vars' function accepting the input, min, max,
    num_bits, and narrow_range values as runtime arguments.
    """

    def __init__(self,
        *,
        custom_fake_quant_op_calls: builtins.bool = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["custom_fake_quant_op_calls",b"custom_fake_quant_op_calls"]) -> None: ...
global___ConversionOptions = ConversionOptions

class Config(google.protobuf.message.Message):
    """Config represents configuration information for tf2xla conversion."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    FEED_FIELD_NUMBER: builtins.int
    FETCH_FIELD_NUMBER: builtins.int
    VARIABLE_FIELD_NUMBER: builtins.int
    CONVERSION_OPTIONS_FIELD_NUMBER: builtins.int
    @property
    def feed(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Feed]:
        """Each feed is a positional input argument for the generated computation.
        The order of each entry matches the order of each input argument.
        """
        pass
    @property
    def fetch(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Fetch]:
        """Each fetch is a positional output argument for the generated computation.
        The order of each entry matches the order of each output argument.
        """
        pass
    @property
    def variable(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Variable]:
        """Each variable is a named input and output of the generated computation."""
        pass
    @property
    def conversion_options(self) -> global___ConversionOptions:
        """Optional conversion options."""
        pass
    def __init__(self,
        *,
        feed: typing.Optional[typing.Iterable[global___Feed]] = ...,
        fetch: typing.Optional[typing.Iterable[global___Fetch]] = ...,
        variable: typing.Optional[typing.Iterable[global___Variable]] = ...,
        conversion_options: typing.Optional[global___ConversionOptions] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["conversion_options",b"conversion_options"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["conversion_options",b"conversion_options","feed",b"feed","fetch",b"fetch","variable",b"variable"]) -> None: ...
global___Config = Config