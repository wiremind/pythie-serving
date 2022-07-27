"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class PaddingMap(google.protobuf.message.Message):
    """A mapping between the dynamic shape dimension of an input and the arg that
    represents the real shape.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ARG_INDEX_FIELD_NUMBER: builtins.int
    SHAPE_INDEX_FIELD_NUMBER: builtins.int
    PADDING_ARG_INDEX_FIELD_NUMBER: builtins.int
    arg_index: builtins.int
    """Input arg index with dynamic shapes."""

    shape_index: builtins.int
    """The dynamic shape dimension index."""

    padding_arg_index: builtins.int
    """The arg index that dynamic dimension maps to, which represents the value
    of the real shape.
    """

    def __init__(self,
        *,
        arg_index: builtins.int = ...,
        shape_index: builtins.int = ...,
        padding_arg_index: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["arg_index",b"arg_index","padding_arg_index",b"padding_arg_index","shape_index",b"shape_index"]) -> None: ...
global___PaddingMap = PaddingMap
