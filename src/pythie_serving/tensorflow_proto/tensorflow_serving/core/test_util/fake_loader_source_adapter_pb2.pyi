"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class FakeLoaderSourceAdapterConfig(google.protobuf.message.Message):
    """Config proto for FakeLoaderSourceAdapter."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SUFFIX_FIELD_NUMBER: builtins.int
    suffix: builtins.str
    """FakeLoaderSourceAdapter's 'suffix' ctor parameter."""
    def __init__(
        self,
        *,
        suffix: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["suffix", b"suffix"]) -> None: ...

global___FakeLoaderSourceAdapterConfig = FakeLoaderSourceAdapterConfig
