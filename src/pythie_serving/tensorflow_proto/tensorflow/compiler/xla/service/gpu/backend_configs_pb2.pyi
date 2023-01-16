"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import tensorflow.compiler.xla.stream_executor.dnn_pb2
import tensorflow.compiler.xla.xla_data_pb2
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class CudnnConvBackendConfig(google.protobuf.message.Message):
    """Backend configs for XLA:GPU.

    These are metadata that the GPU backend attaches to HloInstructions and later
    uses during e.g. codegen.

    Remember that proto3 doesn't give clients a way to tell the difference
    between a field not being present and a field having the default value.
    Choose your defaults carefully.

    No guarantee is made about the stability of these protos.

    See HloInstruction::backend_config() for more info.

    Backend config for a convolution that runs through cudnn.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ALGORITHM_FIELD_NUMBER: builtins.int
    CONV_RESULT_SCALE_FIELD_NUMBER: builtins.int
    ACTIVATION_MODE_FIELD_NUMBER: builtins.int
    SIDE_INPUT_SCALE_FIELD_NUMBER: builtins.int
    @property
    def algorithm(self) -> tensorflow.compiler.xla.stream_executor.dnn_pb2.AlgorithmProto:
        """Opaque algorithm number and tuning knobs chosen for this conv."""
    conv_result_scale: builtins.float
    """The scaling factor multiplied with the convolution result."""
    activation_mode: builtins.int
    """Below are the fields related to cuDNN's fused convolution. Refer to
    GpuConvParams for their meanings.

    The requested activation (e.g. relu) after the convolution. It is with type
    stream_executor::dnn::ActivationMode.
    """
    side_input_scale: builtins.float
    """The scaling factor multiplied with the side input. If no side input buffer
    is provided, this field must be 0.
    """
    def __init__(
        self,
        *,
        algorithm: tensorflow.compiler.xla.stream_executor.dnn_pb2.AlgorithmProto | None = ...,
        conv_result_scale: builtins.float = ...,
        activation_mode: builtins.int = ...,
        side_input_scale: builtins.float = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["algorithm", b"algorithm"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["activation_mode", b"activation_mode", "algorithm", b"algorithm", "conv_result_scale", b"conv_result_scale", "side_input_scale", b"side_input_scale"]) -> None: ...

global___CudnnConvBackendConfig = CudnnConvBackendConfig

@typing_extensions.final
class GemmBackendConfig(google.protobuf.message.Message):
    """Backend config for the GEMM operation running through cuBLAS."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _Epilogue:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _EpilogueEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[GemmBackendConfig._Epilogue.ValueType], builtins.type):  # noqa: F821
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        DEFAULT: GemmBackendConfig._Epilogue.ValueType  # 0
        BIAS: GemmBackendConfig._Epilogue.ValueType  # 1

    class Epilogue(_Epilogue, metaclass=_EpilogueEnumTypeWrapper):
        """cublasLt matmul epilogue."""

    DEFAULT: GemmBackendConfig.Epilogue.ValueType  # 0
    BIAS: GemmBackendConfig.Epilogue.ValueType  # 1

    SELECTED_ALGORITHM_FIELD_NUMBER: builtins.int
    ALPHA_REAL_FIELD_NUMBER: builtins.int
    ALPHA_IMAG_FIELD_NUMBER: builtins.int
    BETA_FIELD_NUMBER: builtins.int
    DOT_DIMENSION_NUMBERS_FIELD_NUMBER: builtins.int
    PRECISION_CONFIG_FIELD_NUMBER: builtins.int
    EPILOGUE_FIELD_NUMBER: builtins.int
    selected_algorithm: builtins.int
    alpha_real: builtins.float
    alpha_imag: builtins.float
    beta: builtins.float
    @property
    def dot_dimension_numbers(self) -> tensorflow.compiler.xla.xla_data_pb2.DotDimensionNumbers: ...
    @property
    def precision_config(self) -> tensorflow.compiler.xla.xla_data_pb2.PrecisionConfig: ...
    epilogue: global___GemmBackendConfig.Epilogue.ValueType
    def __init__(
        self,
        *,
        selected_algorithm: builtins.int = ...,
        alpha_real: builtins.float = ...,
        alpha_imag: builtins.float = ...,
        beta: builtins.float = ...,
        dot_dimension_numbers: tensorflow.compiler.xla.xla_data_pb2.DotDimensionNumbers | None = ...,
        precision_config: tensorflow.compiler.xla.xla_data_pb2.PrecisionConfig | None = ...,
        epilogue: global___GemmBackendConfig.Epilogue.ValueType = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["algorithm", b"algorithm", "dot_dimension_numbers", b"dot_dimension_numbers", "precision_config", b"precision_config", "selected_algorithm", b"selected_algorithm"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["algorithm", b"algorithm", "alpha_imag", b"alpha_imag", "alpha_real", b"alpha_real", "beta", b"beta", "dot_dimension_numbers", b"dot_dimension_numbers", "epilogue", b"epilogue", "precision_config", b"precision_config", "selected_algorithm", b"selected_algorithm"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["algorithm", b"algorithm"]) -> typing_extensions.Literal["selected_algorithm"] | None: ...

global___GemmBackendConfig = GemmBackendConfig

@typing_extensions.final
class BitcastBackendConfig(google.protobuf.message.Message):
    """Backend config for bitcast operation generated from MLIR MHLO dialect."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SOURCE_LAYOUT_FIELD_NUMBER: builtins.int
    RESULT_LAYOUT_FIELD_NUMBER: builtins.int
    @property
    def source_layout(self) -> tensorflow.compiler.xla.xla_data_pb2.LayoutProto: ...
    @property
    def result_layout(self) -> tensorflow.compiler.xla.xla_data_pb2.LayoutProto: ...
    def __init__(
        self,
        *,
        source_layout: tensorflow.compiler.xla.xla_data_pb2.LayoutProto | None = ...,
        result_layout: tensorflow.compiler.xla.xla_data_pb2.LayoutProto | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["result_layout", b"result_layout", "source_layout", b"source_layout"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["result_layout", b"result_layout", "source_layout", b"source_layout"]) -> None: ...

global___BitcastBackendConfig = BitcastBackendConfig
