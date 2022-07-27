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

class Diagnostics(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    INFO_FIELD_NUMBER: builtins.int
    WARNINGS_FIELD_NUMBER: builtins.int
    ERRORS_FIELD_NUMBER: builtins.int
    @property
    def info(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]: ...
    @property
    def warnings(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]: ...
    @property
    def errors(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]: ...
    def __init__(self,
        *,
        info: typing.Optional[typing.Iterable[typing.Text]] = ...,
        warnings: typing.Optional[typing.Iterable[typing.Text]] = ...,
        errors: typing.Optional[typing.Iterable[typing.Text]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["errors",b"errors","info",b"info","warnings",b"warnings"]) -> None: ...
global___Diagnostics = Diagnostics
