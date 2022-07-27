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

class HashmapSourceAdapterConfig(google.protobuf.message.Message):
    """Config proto for HashmapSourceAdapter."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _Format:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _FormatEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[HashmapSourceAdapterConfig._Format.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        SIMPLE_CSV: HashmapSourceAdapterConfig._Format.ValueType  # 0
        """A simple kind of CSV text file of the form:
         key0,value0\\n
         key1,value1\\n
         ...
        """

    class Format(_Format, metaclass=_FormatEnumTypeWrapper):
        """The format used by the file containing a serialized hashmap."""
        pass

    SIMPLE_CSV: HashmapSourceAdapterConfig.Format.ValueType  # 0
    """A simple kind of CSV text file of the form:
     key0,value0\\n
     key1,value1\\n
     ...
    """


    FORMAT_FIELD_NUMBER: builtins.int
    format: global___HashmapSourceAdapterConfig.Format.ValueType
    def __init__(self,
        *,
        format: global___HashmapSourceAdapterConfig.Format.ValueType = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["format",b"format"]) -> None: ...
global___HashmapSourceAdapterConfig = HashmapSourceAdapterConfig
