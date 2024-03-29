"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _QuantizationPrecision:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _QuantizationPrecisionEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_QuantizationPrecision.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    PRECISION_UNSPECIFIED: _QuantizationPrecision.ValueType  # 0
    PRECISION_FULL: _QuantizationPrecision.ValueType  # 1
    """Full Precision (Do not quantize)"""
    PRECISION_W4A4: _QuantizationPrecision.ValueType  # 2
    """Weight 4 bit and activation 4 bit quantization"""
    PRECISION_W4A8: _QuantizationPrecision.ValueType  # 3
    """Weight 4 bit and activation 8 bit quantization"""
    PRECISION_W8A8: _QuantizationPrecision.ValueType  # 4
    """Weight 8 bit and activation 8 bit quantization"""

class QuantizationPrecision(_QuantizationPrecision, metaclass=_QuantizationPrecisionEnumTypeWrapper):
    """Quantization precisions. If the specified quantization
    precision is not available, our quantizer needs to raise an error.
    """

PRECISION_UNSPECIFIED: QuantizationPrecision.ValueType  # 0
PRECISION_FULL: QuantizationPrecision.ValueType  # 1
"""Full Precision (Do not quantize)"""
PRECISION_W4A4: QuantizationPrecision.ValueType  # 2
"""Weight 4 bit and activation 4 bit quantization"""
PRECISION_W4A8: QuantizationPrecision.ValueType  # 3
"""Weight 4 bit and activation 8 bit quantization"""
PRECISION_W8A8: QuantizationPrecision.ValueType  # 4
"""Weight 8 bit and activation 8 bit quantization"""
global___QuantizationPrecision = QuantizationPrecision

class _OpSet:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _OpSetEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_OpSet.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    OP_SET_UNSPECIFIED: _OpSet.ValueType  # 0
    """go/do-include-enum-unspecified"""
    TF: _OpSet.ValueType  # 1
    """Uses TF ops that mimic quantization behavior. Used when the corresponding
    integer op is not yet present.
    """
    XLA: _OpSet.ValueType  # 2
    """Uses TF XLA ops"""
    UNIFORM_QUANTIZED: _OpSet.ValueType  # 3
    """Uses TF Uniform Quantized ops"""

class OpSet(_OpSet, metaclass=_OpSetEnumTypeWrapper):
    """List of supported opsets to deploy the quantized model.
    The quantized model contains different set of ops depending on the opset.
    """

OP_SET_UNSPECIFIED: OpSet.ValueType  # 0
"""go/do-include-enum-unspecified"""
TF: OpSet.ValueType  # 1
"""Uses TF ops that mimic quantization behavior. Used when the corresponding
integer op is not yet present.
"""
XLA: OpSet.ValueType  # 2
"""Uses TF XLA ops"""
UNIFORM_QUANTIZED: OpSet.ValueType  # 3
"""Uses TF Uniform Quantized ops"""
global___OpSet = OpSet

@typing_extensions.final
class QuantizationMethod(google.protobuf.message.Message):
    """TODO(b/240220915): Add a checker for the quantization configuration.
    There will be inconsistencies in the quantization configuration that users
    write. Also, users can write an invalid quantization configuration.
    Therefore, our quantization path will perform validation check for the
    configuration in the future.

    Model quantization method for optimization.

    Various techniques for model quantization are defined within this message
    along with a field that specifies a method to be used for a particular
    quantization request.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _Method:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _MethodEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[QuantizationMethod._Method.ValueType], builtins.type):  # noqa: F821
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        METHOD_UNSPECIFIED: QuantizationMethod._Method.ValueType  # 0
        """This should never be used. Using this will generally result in an error.
        go/do-include-enum-unspecified
        """

    class Method(_Method, metaclass=_MethodEnumTypeWrapper):
        """Quantization methods that are supported as a stable API."""

    METHOD_UNSPECIFIED: QuantizationMethod.Method.ValueType  # 0
    """This should never be used. Using this will generally result in an error.
    go/do-include-enum-unspecified
    """

    class _ExperimentalMethod:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _ExperimentalMethodEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[QuantizationMethod._ExperimentalMethod.ValueType], builtins.type):  # noqa: F821
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        EXPERIMENTAL_METHOD_UNSPECIFIED: QuantizationMethod._ExperimentalMethod.ValueType  # 0
        """This should never be used. Using this will generally result in an error.
        go/do-include-enum-unspecified
        """
        STATIC_RANGE: QuantizationMethod._ExperimentalMethod.ValueType  # 1
        """Static range quantization. Quantized tensor values' ranges are statically
        determined.
        """
        DYNAMIC_RANGE: QuantizationMethod._ExperimentalMethod.ValueType  # 2
        """Dynamic range quantization. Quantized tensor values' ranges are
        determined in the graph executions. The weights are quantized during
        conversion.
        """

    class ExperimentalMethod(_ExperimentalMethod, metaclass=_ExperimentalMethodEnumTypeWrapper):
        """Experimental quantization methods.
        These methods are either not implemented or provided with an unstable
        behavior.
        """

    EXPERIMENTAL_METHOD_UNSPECIFIED: QuantizationMethod.ExperimentalMethod.ValueType  # 0
    """This should never be used. Using this will generally result in an error.
    go/do-include-enum-unspecified
    """
    STATIC_RANGE: QuantizationMethod.ExperimentalMethod.ValueType  # 1
    """Static range quantization. Quantized tensor values' ranges are statically
    determined.
    """
    DYNAMIC_RANGE: QuantizationMethod.ExperimentalMethod.ValueType  # 2
    """Dynamic range quantization. Quantized tensor values' ranges are
    determined in the graph executions. The weights are quantized during
    conversion.
    """

    METHOD_FIELD_NUMBER: builtins.int
    EXPERIMENTAL_METHOD_FIELD_NUMBER: builtins.int
    method: global___QuantizationMethod.Method.ValueType
    experimental_method: global___QuantizationMethod.ExperimentalMethod.ValueType
    def __init__(
        self,
        *,
        method: global___QuantizationMethod.Method.ValueType = ...,
        experimental_method: global___QuantizationMethod.ExperimentalMethod.ValueType = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["experimental_method", b"experimental_method", "method", b"method", "method_oneof", b"method_oneof"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["experimental_method", b"experimental_method", "method", b"method", "method_oneof", b"method_oneof"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["method_oneof", b"method_oneof"]) -> typing_extensions.Literal["method", "experimental_method"] | None: ...

global___QuantizationMethod = QuantizationMethod

@typing_extensions.final
class UnitWiseQuantizationPrecision(google.protobuf.message.Message):
    """Unit (either nodes or ops at this moment) wise quantization method for
    mixed bit precision quantization. It contains the name of the unit,
    the granularity of the unit, and the quantization method for each unit.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _UnitType:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _UnitTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[UnitWiseQuantizationPrecision._UnitType.ValueType], builtins.type):  # noqa: F821
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        UNIT_UNSPECIFIED: UnitWiseQuantizationPrecision._UnitType.ValueType  # 0
        """This should never be used. Using this will generally result in an error."""
        UNIT_NODE: UnitWiseQuantizationPrecision._UnitType.ValueType  # 1
        UNIT_OP: UnitWiseQuantizationPrecision._UnitType.ValueType  # 2

    class UnitType(_UnitType, metaclass=_UnitTypeEnumTypeWrapper):
        """Quantization unit granularity."""

    UNIT_UNSPECIFIED: UnitWiseQuantizationPrecision.UnitType.ValueType  # 0
    """This should never be used. Using this will generally result in an error."""
    UNIT_NODE: UnitWiseQuantizationPrecision.UnitType.ValueType  # 1
    UNIT_OP: UnitWiseQuantizationPrecision.UnitType.ValueType  # 2

    UNIT_TYPE_FIELD_NUMBER: builtins.int
    FUNC_NAME_FIELD_NUMBER: builtins.int
    UNIT_NAME_FIELD_NUMBER: builtins.int
    QUANTIZATION_PRECISION_FIELD_NUMBER: builtins.int
    unit_type: global___UnitWiseQuantizationPrecision.UnitType.ValueType
    """Available quantization unit. Currently node-wise and op-wise are
    available quantization units.
    """
    func_name: builtins.str
    """Uniqueness isn't guaranteed across SavedModels but within each function
    def's level, uniqueness is guaranteed. Updated
    the configuration interfaces to reflect such circumstances.
    If users do not need to guarantee uniqueness func_name can be omitted.
    """
    unit_name: builtins.str
    quantization_precision: global___QuantizationPrecision.ValueType
    """Quantization option information for the current unit.
    TODO(b/241322587): Support specifying quantization method for each unit of
    TF GraphDef.
    """
    def __init__(
        self,
        *,
        unit_type: global___UnitWiseQuantizationPrecision.UnitType.ValueType = ...,
        func_name: builtins.str = ...,
        unit_name: builtins.str = ...,
        quantization_precision: global___QuantizationPrecision.ValueType = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["func_name", b"func_name", "quantization_precision", b"quantization_precision", "unit_name", b"unit_name", "unit_type", b"unit_type"]) -> None: ...

global___UnitWiseQuantizationPrecision = UnitWiseQuantizationPrecision

@typing_extensions.final
class QuantizationOptions(google.protobuf.message.Message):
    """Defines various options to specify and control the behavior of the quantizer.
    It consists of
    1) Model-wise quantization configuration as a default configuration. If it is
    None, the default configuration is "do not quantize the model".
    2) A set of supported operations.
    3) Unit wise quantization precision.
    4) Target hardware name.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    QUANTIZATION_METHOD_FIELD_NUMBER: builtins.int
    OP_SET_FIELD_NUMBER: builtins.int
    QUANTIZATION_PRECISION_FIELD_NUMBER: builtins.int
    UNIT_WISE_QUANTIZATION_PRECISION_FIELD_NUMBER: builtins.int
    MIN_NUM_ELEMENTS_FOR_WEIGHTS_FIELD_NUMBER: builtins.int
    @property
    def quantization_method(self) -> global___QuantizationMethod:
        """The default quantization configuration for the model. If the below
        unit-wise configuration does not exist, we use this default quantization
        configuration for the entire model. If the below unit-wise configuration
        exists, this default one will become the quantization configuration for
        units that are not specified in unit-wise configurations.
        """
    op_set: global___OpSet.ValueType
    quantization_precision: global___QuantizationPrecision.ValueType
    @property
    def unit_wise_quantization_precision(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___UnitWiseQuantizationPrecision]:
        """Quantization precision for each unit. Units can become either
        nodes or ops, and the mixture of those different units are allowed.
        If there are conflicts or ambiguity in this unit-wise precision, our
        quantizer will raise an error.
        """
    min_num_elements_for_weights: builtins.int
    """Minimum number of weight elements to apply quantization. Currently only
    supported for Post-training Dynamic Range Quantization. By default, it is
    set to 1024. To disable this, set the value to -1 explicitly.
    """
    def __init__(
        self,
        *,
        quantization_method: global___QuantizationMethod | None = ...,
        op_set: global___OpSet.ValueType = ...,
        quantization_precision: global___QuantizationPrecision.ValueType = ...,
        unit_wise_quantization_precision: collections.abc.Iterable[global___UnitWiseQuantizationPrecision] | None = ...,
        min_num_elements_for_weights: builtins.int = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["quantization_method", b"quantization_method"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["min_num_elements_for_weights", b"min_num_elements_for_weights", "op_set", b"op_set", "quantization_method", b"quantization_method", "quantization_precision", b"quantization_precision", "unit_wise_quantization_precision", b"unit_wise_quantization_precision"]) -> None: ...

global___QuantizationOptions = QuantizationOptions
