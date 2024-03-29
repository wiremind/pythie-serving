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
import tensorflow.core.framework.tensor_pb2
import tensorflow.core.protobuf.struct_pb2

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class CallRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    METHOD_FIELD_NUMBER: builtins.int
    INPUT_TENSORS_FIELD_NUMBER: builtins.int
    method: builtins.str
    @property
    def input_tensors(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tensorflow.core.framework.tensor_pb2.TensorProto]: ...
    def __init__(
        self,
        *,
        method: builtins.str = ...,
        input_tensors: collections.abc.Iterable[tensorflow.core.framework.tensor_pb2.TensorProto] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["input_tensors", b"input_tensors", "method", b"method"]) -> None: ...

global___CallRequest = CallRequest

@typing_extensions.final
class CallResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    OUTPUT_TENSORS_FIELD_NUMBER: builtins.int
    @property
    def output_tensors(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tensorflow.core.framework.tensor_pb2.TensorProto]: ...
    def __init__(
        self,
        *,
        output_tensors: collections.abc.Iterable[tensorflow.core.framework.tensor_pb2.TensorProto] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["output_tensors", b"output_tensors"]) -> None: ...

global___CallResponse = CallResponse

@typing_extensions.final
class ListRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___ListRequest = ListRequest

@typing_extensions.final
class RegisteredMethod(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    METHOD_FIELD_NUMBER: builtins.int
    INPUT_SPECS_FIELD_NUMBER: builtins.int
    OUTPUT_SPECS_FIELD_NUMBER: builtins.int
    method: builtins.str
    @property
    def input_specs(self) -> tensorflow.core.protobuf.struct_pb2.StructuredValue: ...
    @property
    def output_specs(self) -> tensorflow.core.protobuf.struct_pb2.StructuredValue: ...
    def __init__(
        self,
        *,
        method: builtins.str = ...,
        input_specs: tensorflow.core.protobuf.struct_pb2.StructuredValue | None = ...,
        output_specs: tensorflow.core.protobuf.struct_pb2.StructuredValue | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["input_specs", b"input_specs", "output_specs", b"output_specs"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["input_specs", b"input_specs", "method", b"method", "output_specs", b"output_specs"]) -> None: ...

global___RegisteredMethod = RegisteredMethod

@typing_extensions.final
class ListResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    REGISTERED_METHODS_FIELD_NUMBER: builtins.int
    @property
    def registered_methods(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RegisteredMethod]: ...
    def __init__(
        self,
        *,
        registered_methods: collections.abc.Iterable[global___RegisteredMethod] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["registered_methods", b"registered_methods"]) -> None: ...

global___ListResponse = ListResponse
