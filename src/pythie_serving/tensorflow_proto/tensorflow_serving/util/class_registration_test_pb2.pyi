"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.any_pb2
import google.protobuf.descriptor
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Config1(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    STRING_FIELD_FIELD_NUMBER: builtins.int
    string_field: typing.Text
    def __init__(self,
        *,
        string_field: typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["string_field",b"string_field"]) -> None: ...
global___Config1 = Config1

class Config2(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    STRING_FIELD_FIELD_NUMBER: builtins.int
    string_field: typing.Text
    def __init__(self,
        *,
        string_field: typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["string_field",b"string_field"]) -> None: ...
global___Config2 = Config2

class MessageWithAny(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ANY_FIELD_FIELD_NUMBER: builtins.int
    @property
    def any_field(self) -> google.protobuf.any_pb2.Any: ...
    def __init__(self,
        *,
        any_field: typing.Optional[google.protobuf.any_pb2.Any] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["any_field",b"any_field"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["any_field",b"any_field"]) -> None: ...
global___MessageWithAny = MessageWithAny