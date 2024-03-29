"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
This proto intends to match format expected by pprof tool."""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class Profile(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SAMPLE_TYPE_FIELD_NUMBER: builtins.int
    SAMPLE_FIELD_NUMBER: builtins.int
    MAPPING_FIELD_NUMBER: builtins.int
    LOCATION_FIELD_NUMBER: builtins.int
    FUNCTION_FIELD_NUMBER: builtins.int
    STRING_TABLE_FIELD_NUMBER: builtins.int
    DROP_FRAMES_FIELD_NUMBER: builtins.int
    KEEP_FRAMES_FIELD_NUMBER: builtins.int
    TIME_NANOS_FIELD_NUMBER: builtins.int
    DURATION_NANOS_FIELD_NUMBER: builtins.int
    PERIOD_TYPE_FIELD_NUMBER: builtins.int
    PERIOD_FIELD_NUMBER: builtins.int
    COMMENT_FIELD_NUMBER: builtins.int
    DEFAULT_SAMPLE_TYPE_FIELD_NUMBER: builtins.int
    @property
    def sample_type(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ValueType]: ...
    @property
    def sample(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Sample]: ...
    @property
    def mapping(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Mapping]: ...
    @property
    def location(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Location]: ...
    @property
    def function(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Function]: ...
    @property
    def string_table(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    drop_frames: builtins.int
    keep_frames: builtins.int
    time_nanos: builtins.int
    duration_nanos: builtins.int
    @property
    def period_type(self) -> global___ValueType: ...
    period: builtins.int
    @property
    def comment(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    default_sample_type: builtins.int
    def __init__(
        self,
        *,
        sample_type: collections.abc.Iterable[global___ValueType] | None = ...,
        sample: collections.abc.Iterable[global___Sample] | None = ...,
        mapping: collections.abc.Iterable[global___Mapping] | None = ...,
        location: collections.abc.Iterable[global___Location] | None = ...,
        function: collections.abc.Iterable[global___Function] | None = ...,
        string_table: collections.abc.Iterable[builtins.str] | None = ...,
        drop_frames: builtins.int = ...,
        keep_frames: builtins.int = ...,
        time_nanos: builtins.int = ...,
        duration_nanos: builtins.int = ...,
        period_type: global___ValueType | None = ...,
        period: builtins.int = ...,
        comment: collections.abc.Iterable[builtins.int] | None = ...,
        default_sample_type: builtins.int = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["period_type", b"period_type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["comment", b"comment", "default_sample_type", b"default_sample_type", "drop_frames", b"drop_frames", "duration_nanos", b"duration_nanos", "function", b"function", "keep_frames", b"keep_frames", "location", b"location", "mapping", b"mapping", "period", b"period", "period_type", b"period_type", "sample", b"sample", "sample_type", b"sample_type", "string_table", b"string_table", "time_nanos", b"time_nanos"]) -> None: ...

global___Profile = Profile

@typing_extensions.final
class ValueType(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TYPE_FIELD_NUMBER: builtins.int
    UNIT_FIELD_NUMBER: builtins.int
    type: builtins.int
    unit: builtins.int
    def __init__(
        self,
        *,
        type: builtins.int = ...,
        unit: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["type", b"type", "unit", b"unit"]) -> None: ...

global___ValueType = ValueType

@typing_extensions.final
class Sample(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LOCATION_ID_FIELD_NUMBER: builtins.int
    VALUE_FIELD_NUMBER: builtins.int
    LABEL_FIELD_NUMBER: builtins.int
    @property
    def location_id(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def value(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def label(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Label]: ...
    def __init__(
        self,
        *,
        location_id: collections.abc.Iterable[builtins.int] | None = ...,
        value: collections.abc.Iterable[builtins.int] | None = ...,
        label: collections.abc.Iterable[global___Label] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["label", b"label", "location_id", b"location_id", "value", b"value"]) -> None: ...

global___Sample = Sample

@typing_extensions.final
class Label(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    KEY_FIELD_NUMBER: builtins.int
    STR_FIELD_NUMBER: builtins.int
    NUM_FIELD_NUMBER: builtins.int
    key: builtins.int
    str: builtins.int
    num: builtins.int
    def __init__(
        self,
        *,
        key: builtins.int = ...,
        str: builtins.int = ...,
        num: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "num", b"num", "str", b"str"]) -> None: ...

global___Label = Label

@typing_extensions.final
class Mapping(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    MEMORY_START_FIELD_NUMBER: builtins.int
    MEMORY_LIMIT_FIELD_NUMBER: builtins.int
    FILE_OFFSET_FIELD_NUMBER: builtins.int
    FILENAME_FIELD_NUMBER: builtins.int
    BUILD_ID_FIELD_NUMBER: builtins.int
    HAS_FUNCTIONS_FIELD_NUMBER: builtins.int
    HAS_FILENAMES_FIELD_NUMBER: builtins.int
    HAS_LINE_NUMBERS_FIELD_NUMBER: builtins.int
    HAS_INLINE_FRAMES_FIELD_NUMBER: builtins.int
    id: builtins.int
    memory_start: builtins.int
    memory_limit: builtins.int
    file_offset: builtins.int
    filename: builtins.int
    build_id: builtins.int
    has_functions: builtins.bool
    has_filenames: builtins.bool
    has_line_numbers: builtins.bool
    has_inline_frames: builtins.bool
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        memory_start: builtins.int = ...,
        memory_limit: builtins.int = ...,
        file_offset: builtins.int = ...,
        filename: builtins.int = ...,
        build_id: builtins.int = ...,
        has_functions: builtins.bool = ...,
        has_filenames: builtins.bool = ...,
        has_line_numbers: builtins.bool = ...,
        has_inline_frames: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["build_id", b"build_id", "file_offset", b"file_offset", "filename", b"filename", "has_filenames", b"has_filenames", "has_functions", b"has_functions", "has_inline_frames", b"has_inline_frames", "has_line_numbers", b"has_line_numbers", "id", b"id", "memory_limit", b"memory_limit", "memory_start", b"memory_start"]) -> None: ...

global___Mapping = Mapping

@typing_extensions.final
class Location(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    MAPPING_ID_FIELD_NUMBER: builtins.int
    ADDRESS_FIELD_NUMBER: builtins.int
    LINE_FIELD_NUMBER: builtins.int
    id: builtins.int
    mapping_id: builtins.int
    address: builtins.int
    @property
    def line(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Line]: ...
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        mapping_id: builtins.int = ...,
        address: builtins.int = ...,
        line: collections.abc.Iterable[global___Line] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["address", b"address", "id", b"id", "line", b"line", "mapping_id", b"mapping_id"]) -> None: ...

global___Location = Location

@typing_extensions.final
class Line(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    FUNCTION_ID_FIELD_NUMBER: builtins.int
    LINE_FIELD_NUMBER: builtins.int
    function_id: builtins.int
    line: builtins.int
    def __init__(
        self,
        *,
        function_id: builtins.int = ...,
        line: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["function_id", b"function_id", "line", b"line"]) -> None: ...

global___Line = Line

@typing_extensions.final
class Function(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    SYSTEM_NAME_FIELD_NUMBER: builtins.int
    FILENAME_FIELD_NUMBER: builtins.int
    START_LINE_FIELD_NUMBER: builtins.int
    id: builtins.int
    name: builtins.int
    system_name: builtins.int
    filename: builtins.int
    start_line: builtins.int
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        name: builtins.int = ...,
        system_name: builtins.int = ...,
        filename: builtins.int = ...,
        start_line: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["filename", b"filename", "id", b"id", "name", b"name", "start_line", b"start_line", "system_name", b"system_name"]) -> None: ...

global___Function = Function
