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

class BytesList(google.protobuf.message.Message):
    """LINT.IfChange
    Containers to hold repeated fundamental values.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    VALUE_FIELD_NUMBER: builtins.int
    @property
    def value(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bytes]: ...
    def __init__(self,
        *,
        value: typing.Optional[typing.Iterable[builtins.bytes]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["value",b"value"]) -> None: ...
global___BytesList = BytesList

class FloatList(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    VALUE_FIELD_NUMBER: builtins.int
    @property
    def value(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]: ...
    def __init__(self,
        *,
        value: typing.Optional[typing.Iterable[builtins.float]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["value",b"value"]) -> None: ...
global___FloatList = FloatList

class Int64List(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    VALUE_FIELD_NUMBER: builtins.int
    @property
    def value(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(self,
        *,
        value: typing.Optional[typing.Iterable[builtins.int]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["value",b"value"]) -> None: ...
global___Int64List = Int64List

class Feature(google.protobuf.message.Message):
    """Containers for non-sequential data."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    BYTES_LIST_FIELD_NUMBER: builtins.int
    FLOAT_LIST_FIELD_NUMBER: builtins.int
    INT64_LIST_FIELD_NUMBER: builtins.int
    @property
    def bytes_list(self) -> global___BytesList: ...
    @property
    def float_list(self) -> global___FloatList: ...
    @property
    def int64_list(self) -> global___Int64List: ...
    def __init__(self,
        *,
        bytes_list: typing.Optional[global___BytesList] = ...,
        float_list: typing.Optional[global___FloatList] = ...,
        int64_list: typing.Optional[global___Int64List] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["bytes_list",b"bytes_list","float_list",b"float_list","int64_list",b"int64_list","kind",b"kind"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["bytes_list",b"bytes_list","float_list",b"float_list","int64_list",b"int64_list","kind",b"kind"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["kind",b"kind"]) -> typing.Optional[typing_extensions.Literal["bytes_list","float_list","int64_list"]]: ...
global___Feature = Feature

class Features(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class FeatureEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> global___Feature: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[global___Feature] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    FEATURE_FIELD_NUMBER: builtins.int
    @property
    def feature(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___Feature]:
        """Map from feature name to feature."""
        pass
    def __init__(self,
        *,
        feature: typing.Optional[typing.Mapping[typing.Text, global___Feature]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["feature",b"feature"]) -> None: ...
global___Features = Features

class FeatureList(google.protobuf.message.Message):
    """Containers for sequential data.

    A FeatureList contains lists of Features.  These may hold zero or more
    Feature values.

    FeatureLists are organized into categories by name.  The FeatureLists message
    contains the mapping from name to FeatureList.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    FEATURE_FIELD_NUMBER: builtins.int
    @property
    def feature(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Feature]: ...
    def __init__(self,
        *,
        feature: typing.Optional[typing.Iterable[global___Feature]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["feature",b"feature"]) -> None: ...
global___FeatureList = FeatureList

class FeatureLists(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class FeatureListEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> global___FeatureList: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[global___FeatureList] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    FEATURE_LIST_FIELD_NUMBER: builtins.int
    @property
    def feature_list(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___FeatureList]:
        """Map from feature name to feature list."""
        pass
    def __init__(self,
        *,
        feature_list: typing.Optional[typing.Mapping[typing.Text, global___FeatureList]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["feature_list",b"feature_list"]) -> None: ...
global___FeatureLists = FeatureLists