"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class VerifierConfig(google.protobuf.message.Message):
    """The config for graph verifiers."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _Toggle:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _ToggleEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[VerifierConfig._Toggle.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        DEFAULT: VerifierConfig._Toggle.ValueType  # 0
        ON: VerifierConfig._Toggle.ValueType  # 1
        OFF: VerifierConfig._Toggle.ValueType  # 2
    class Toggle(_Toggle, metaclass=_ToggleEnumTypeWrapper):
        pass

    DEFAULT: VerifierConfig.Toggle.ValueType  # 0
    ON: VerifierConfig.Toggle.ValueType  # 1
    OFF: VerifierConfig.Toggle.ValueType  # 2

    VERIFICATION_TIMEOUT_IN_MS_FIELD_NUMBER: builtins.int
    STRUCTURE_VERIFIER_FIELD_NUMBER: builtins.int
    verification_timeout_in_ms: builtins.int
    """Deadline for completion of all verification i.e. all the Toggle ON
    verifiers must complete execution within this time.
    """

    structure_verifier: global___VerifierConfig.Toggle.ValueType
    """Perform structural validation on a tensorflow graph. Default is OFF."""

    def __init__(self,
        *,
        verification_timeout_in_ms: builtins.int = ...,
        structure_verifier: global___VerifierConfig.Toggle.ValueType = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["structure_verifier",b"structure_verifier","verification_timeout_in_ms",b"verification_timeout_in_ms"]) -> None: ...
global___VerifierConfig = VerifierConfig