"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys
import tensorflow.compiler.xla.xla_data_pb2

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class RunHloModuleIterationLiterals(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ARGUMENTS_FIELD_NUMBER: builtins.int
    RESULT_FIELD_NUMBER: builtins.int
    REFERENCE_RESULT_FIELD_NUMBER: builtins.int
    @property
    def arguments(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tensorflow.compiler.xla.xla_data_pb2.LiteralProto]:
        """Arguments used by the iteration."""
    @property
    def result(self) -> tensorflow.compiler.xla.xla_data_pb2.LiteralProto:
        """Ressult of the iteration on the target platform."""
    @property
    def reference_result(self) -> tensorflow.compiler.xla.xla_data_pb2.LiteralProto:
        """Result of the iteration on the reference platform."""
    def __init__(
        self,
        *,
        arguments: collections.abc.Iterable[tensorflow.compiler.xla.xla_data_pb2.LiteralProto] | None = ...,
        result: tensorflow.compiler.xla.xla_data_pb2.LiteralProto | None = ...,
        reference_result: tensorflow.compiler.xla.xla_data_pb2.LiteralProto | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["reference_result", b"reference_result", "result", b"result"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["arguments", b"arguments", "reference_result", b"reference_result", "result", b"result"]) -> None: ...

global___RunHloModuleIterationLiterals = RunHloModuleIterationLiterals

@typing_extensions.final
class RunHloModuleLiterals(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ITERATIONS_FIELD_NUMBER: builtins.int
    @property
    def iterations(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RunHloModuleIterationLiterals]:
        """Iterations of run hlo module."""
    def __init__(
        self,
        *,
        iterations: collections.abc.Iterable[global___RunHloModuleIterationLiterals] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["iterations", b"iterations"]) -> None: ...

global___RunHloModuleLiterals = RunHloModuleLiterals
