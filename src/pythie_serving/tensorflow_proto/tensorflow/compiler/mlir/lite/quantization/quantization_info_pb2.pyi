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

class QuantizationInfo(google.protobuf.message.Message):
    """Represents the quantization parameters for a list of named tensors."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class MinMax(google.protobuf.message.Message):
        """min/max of the per axis value range. To quantize the value, the metadata
        of the target properties should be specified or read from the ops
        quantization specification.
        """
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        MIN_FIELD_NUMBER: builtins.int
        MAX_FIELD_NUMBER: builtins.int
        min: builtins.float
        max: builtins.float
        def __init__(self,
            *,
            min: builtins.float = ...,
            max: builtins.float = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["max",b"max","min",b"min"]) -> None: ...

    class AffineParams(google.protobuf.message.Message):
        """Affine parameters to quantize the per axis value. The metadata of the
        target properties should be specified as well.
        """
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        SCALE_FIELD_NUMBER: builtins.int
        ZERO_POINT_FIELD_NUMBER: builtins.int
        scale: builtins.float
        zero_point: builtins.int
        def __init__(self,
            *,
            scale: builtins.float = ...,
            zero_point: builtins.int = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["scale",b"scale","zero_point",b"zero_point"]) -> None: ...

    class PerAxisParams(google.protobuf.message.Message):
        """Params to quantize the axis. Only one of the field can be used."""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        MIN_MAX_FIELD_NUMBER: builtins.int
        AFFINE_PARAMS_FIELD_NUMBER: builtins.int
        @property
        def min_max(self) -> global___QuantizationInfo.MinMax:
            """min/max of the ranges."""
            pass
        @property
        def affine_params(self) -> global___QuantizationInfo.AffineParams:
            """affine parameters to quantize the per axis value."""
            pass
        def __init__(self,
            *,
            min_max: typing.Optional[global___QuantizationInfo.MinMax] = ...,
            affine_params: typing.Optional[global___QuantizationInfo.AffineParams] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["affine_params",b"affine_params","min_max",b"min_max","params_oneof",b"params_oneof"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["affine_params",b"affine_params","min_max",b"min_max","params_oneof",b"params_oneof"]) -> None: ...
        def WhichOneof(self, oneof_group: typing_extensions.Literal["params_oneof",b"params_oneof"]) -> typing.Optional[typing_extensions.Literal["min_max","affine_params"]]: ...

    class Metadata(google.protobuf.message.Message):
        """The metadata defines the target properties."""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        NUM_BITS_FIELD_NUMBER: builtins.int
        QUANTIZE_AXIS_FIELD_NUMBER: builtins.int
        RANGE_MIN_FIELD_NUMBER: builtins.int
        RANGE_MAX_FIELD_NUMBER: builtins.int
        num_bits: builtins.int
        """ Bit number of fixed-point data the target kernel supports."""

        quantize_axis: builtins.int
        """ The quantized axis index if it is per-axis quantization."""

        range_min: builtins.int
        """The minimum allowed value of the fixed-point data range.
        This can also be used to derive the sign of storage type.
        """

        range_max: builtins.int
        """The minimum allowed value of the fixed-point data range."""

        def __init__(self,
            *,
            num_bits: builtins.int = ...,
            quantize_axis: builtins.int = ...,
            range_min: builtins.int = ...,
            range_max: builtins.int = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["num_bits",b"num_bits","quantize_axis",b"quantize_axis","range_max",b"range_max","range_min",b"range_min"]) -> None: ...

    class QuantParams(google.protobuf.message.Message):
        """The quantization parameters for a named tensor."""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        NAME_FIELD_NUMBER: builtins.int
        NAME_REGEX_FIELD_NUMBER: builtins.int
        PARAMS_FIELD_NUMBER: builtins.int
        META_FIELD_NUMBER: builtins.int
        name: typing.Text
        name_regex: typing.Text
        """An regex can be used to match multiple tensors."""

        @property
        def params(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___QuantizationInfo.PerAxisParams]:
            """The quantization parameters for the tensor. If it is for per-axis, the
            parameters should be defined for each axis, otherwise, if it is for
            per-tensor, this repeated field should only contain a single element.
            """
            pass
        @property
        def meta(self) -> global___QuantizationInfo.Metadata:
            """Metadata about the quantization parameters."""
            pass
        def __init__(self,
            *,
            name: typing.Text = ...,
            name_regex: typing.Text = ...,
            params: typing.Optional[typing.Iterable[global___QuantizationInfo.PerAxisParams]] = ...,
            meta: typing.Optional[global___QuantizationInfo.Metadata] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["meta",b"meta","name",b"name","name_oneof",b"name_oneof","name_regex",b"name_regex"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["meta",b"meta","name",b"name","name_oneof",b"name_oneof","name_regex",b"name_regex","params",b"params"]) -> None: ...
        def WhichOneof(self, oneof_group: typing_extensions.Literal["name_oneof",b"name_oneof"]) -> typing.Optional[typing_extensions.Literal["name","name_regex"]]: ...

    ENTRIES_FIELD_NUMBER: builtins.int
    @property
    def entries(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___QuantizationInfo.QuantParams]:
        """List of quantization parameters for tensors."""
        pass
    def __init__(self,
        *,
        entries: typing.Optional[typing.Iterable[global___QuantizationInfo.QuantParams]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["entries",b"entries"]) -> None: ...
global___QuantizationInfo = QuantizationInfo