"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import tensorflow.stream_executor.dnn_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class ConvolutionProto(google.protobuf.message.Message):
    """A convolution. Currently it's only used for logging. In the future, we may
    want to use it in the API as well.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    KIND_FIELD_NUMBER: builtins.int
    INPUT_FIELD_NUMBER: builtins.int
    FILTER_FIELD_NUMBER: builtins.int
    OUTPUT_FIELD_NUMBER: builtins.int
    CONV_DESC_FIELD_NUMBER: builtins.int
    CONV_SCALE_FIELD_NUMBER: builtins.int
    SIDE_VALUE_SCALE_FIELD_NUMBER: builtins.int
    ACTIVATION_FIELD_NUMBER: builtins.int
    INPUT_ADDRESS_FIELD_NUMBER: builtins.int
    FILTER_ADDRESS_FIELD_NUMBER: builtins.int
    OUTPUT_ADDRESS_FIELD_NUMBER: builtins.int
    BIAS_ADDRESS_FIELD_NUMBER: builtins.int
    SIDE_INPUT_ADDRESS_FIELD_NUMBER: builtins.int
    kind: tensorflow.stream_executor.dnn_pb2.ConvolutionKind.ValueType
    @property
    def input(self) -> tensorflow.stream_executor.dnn_pb2.TensorDescriptorProto: ...
    @property
    def filter(self) -> tensorflow.stream_executor.dnn_pb2.TensorDescriptorProto: ...
    @property
    def output(self) -> tensorflow.stream_executor.dnn_pb2.TensorDescriptorProto: ...
    @property
    def conv_desc(self) -> tensorflow.stream_executor.dnn_pb2.ConvolutionDescriptorProto: ...
    conv_scale: builtins.float
    """result = conv_scale * conv(...) + side_value_scale * side_value.
    side_value is an arbitrary buffer if activation is not none. Otherwise, it
    has to be the result buffer (using its old values).
    """

    side_value_scale: builtins.float
    activation: tensorflow.stream_executor.dnn_pb2.ActivationMode.ValueType
    input_address: builtins.int
    filter_address: builtins.int
    output_address: builtins.int
    bias_address: builtins.int
    side_input_address: builtins.int
    def __init__(self,
        *,
        kind: tensorflow.stream_executor.dnn_pb2.ConvolutionKind.ValueType = ...,
        input: typing.Optional[tensorflow.stream_executor.dnn_pb2.TensorDescriptorProto] = ...,
        filter: typing.Optional[tensorflow.stream_executor.dnn_pb2.TensorDescriptorProto] = ...,
        output: typing.Optional[tensorflow.stream_executor.dnn_pb2.TensorDescriptorProto] = ...,
        conv_desc: typing.Optional[tensorflow.stream_executor.dnn_pb2.ConvolutionDescriptorProto] = ...,
        conv_scale: builtins.float = ...,
        side_value_scale: builtins.float = ...,
        activation: tensorflow.stream_executor.dnn_pb2.ActivationMode.ValueType = ...,
        input_address: builtins.int = ...,
        filter_address: builtins.int = ...,
        output_address: builtins.int = ...,
        bias_address: builtins.int = ...,
        side_input_address: builtins.int = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["conv_desc",b"conv_desc","filter",b"filter","input",b"input","output",b"output"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["activation",b"activation","bias_address",b"bias_address","conv_desc",b"conv_desc","conv_scale",b"conv_scale","filter",b"filter","filter_address",b"filter_address","input",b"input","input_address",b"input_address","kind",b"kind","output",b"output","output_address",b"output_address","side_input_address",b"side_input_address","side_value_scale",b"side_value_scale"]) -> None: ...
global___ConvolutionProto = ConvolutionProto