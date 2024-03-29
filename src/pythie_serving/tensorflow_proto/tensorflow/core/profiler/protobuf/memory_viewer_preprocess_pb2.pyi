"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Protobuf definitions for communicating the results of the memory
visualization analysis subprocess (written in C++) to the outer script which
renders HTML from Python.
"""
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
class HeapObject(google.protobuf.message.Message):
    """Describes a heap object that is displayed in a plot in the memory
    visualization HTML.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUMBERED_FIELD_NUMBER: builtins.int
    NAMED_FIELD_NUMBER: builtins.int
    LABEL_FIELD_NUMBER: builtins.int
    LOGICAL_BUFFER_ID_FIELD_NUMBER: builtins.int
    LOGICAL_BUFFER_SIZE_MIB_FIELD_NUMBER: builtins.int
    UNPADDED_SHAPE_MIB_FIELD_NUMBER: builtins.int
    INSTRUCTION_NAME_FIELD_NUMBER: builtins.int
    SHAPE_STRING_FIELD_NUMBER: builtins.int
    TF_OP_NAME_FIELD_NUMBER: builtins.int
    GROUP_NAME_FIELD_NUMBER: builtins.int
    OP_CODE_FIELD_NUMBER: builtins.int
    numbered: builtins.int
    named: builtins.str
    label: builtins.str
    logical_buffer_id: builtins.int
    logical_buffer_size_mib: builtins.float
    unpadded_shape_mib: builtins.float
    instruction_name: builtins.str
    shape_string: builtins.str
    tf_op_name: builtins.str
    group_name: builtins.str
    op_code: builtins.str
    def __init__(
        self,
        *,
        numbered: builtins.int = ...,
        named: builtins.str = ...,
        label: builtins.str = ...,
        logical_buffer_id: builtins.int = ...,
        logical_buffer_size_mib: builtins.float = ...,
        unpadded_shape_mib: builtins.float = ...,
        instruction_name: builtins.str = ...,
        shape_string: builtins.str = ...,
        tf_op_name: builtins.str = ...,
        group_name: builtins.str = ...,
        op_code: builtins.str = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["color", b"color", "named", b"named", "numbered", b"numbered"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["color", b"color", "group_name", b"group_name", "instruction_name", b"instruction_name", "label", b"label", "logical_buffer_id", b"logical_buffer_id", "logical_buffer_size_mib", b"logical_buffer_size_mib", "named", b"named", "numbered", b"numbered", "op_code", b"op_code", "shape_string", b"shape_string", "tf_op_name", b"tf_op_name", "unpadded_shape_mib", b"unpadded_shape_mib"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["color", b"color"]) -> typing_extensions.Literal["numbered", "named"] | None: ...

global___HeapObject = HeapObject

@typing_extensions.final
class BufferSpan(google.protobuf.message.Message):
    """Describes the start / exclusive limit HLO program points for a given buffer
    lifetime, used for rendering a box on the plot.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    START_FIELD_NUMBER: builtins.int
    LIMIT_FIELD_NUMBER: builtins.int
    start: builtins.int
    limit: builtins.int
    def __init__(
        self,
        *,
        start: builtins.int = ...,
        limit: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["limit", b"limit", "start", b"start"]) -> None: ...

global___BufferSpan = BufferSpan

@typing_extensions.final
class LogicalBuffer(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    SHAPE_FIELD_NUMBER: builtins.int
    SIZE_MIB_FIELD_NUMBER: builtins.int
    HLO_NAME_FIELD_NUMBER: builtins.int
    SHAPE_INDEX_FIELD_NUMBER: builtins.int
    id: builtins.int
    shape: builtins.str
    size_mib: builtins.float
    hlo_name: builtins.str
    @property
    def shape_index(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        shape: builtins.str = ...,
        size_mib: builtins.float = ...,
        hlo_name: builtins.str = ...,
        shape_index: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["hlo_name", b"hlo_name", "id", b"id", "shape", b"shape", "shape_index", b"shape_index", "size_mib", b"size_mib"]) -> None: ...

global___LogicalBuffer = LogicalBuffer

@typing_extensions.final
class BufferAllocation(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    SIZE_MIB_FIELD_NUMBER: builtins.int
    ATTRIBUTES_FIELD_NUMBER: builtins.int
    LOGICAL_BUFFERS_FIELD_NUMBER: builtins.int
    COMMON_SHAPE_FIELD_NUMBER: builtins.int
    id: builtins.int
    size_mib: builtins.float
    @property
    def attributes(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    @property
    def logical_buffers(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___LogicalBuffer]: ...
    common_shape: builtins.str
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        size_mib: builtins.float = ...,
        attributes: collections.abc.Iterable[builtins.str] | None = ...,
        logical_buffers: collections.abc.Iterable[global___LogicalBuffer] | None = ...,
        common_shape: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["attributes", b"attributes", "common_shape", b"common_shape", "id", b"id", "logical_buffers", b"logical_buffers", "size_mib", b"size_mib"]) -> None: ...

global___BufferAllocation = BufferAllocation

@typing_extensions.final
class PreprocessResult(google.protobuf.message.Message):
    """Groups together all results from the preprocessing C++ step."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class LogicalBufferSpansEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.int
        @property
        def value(self) -> global___BufferSpan: ...
        def __init__(
            self,
            *,
            key: builtins.int = ...,
            value: global___BufferSpan | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    HEAP_SIZES_FIELD_NUMBER: builtins.int
    UNPADDED_HEAP_SIZES_FIELD_NUMBER: builtins.int
    MAX_HEAP_FIELD_NUMBER: builtins.int
    MAX_HEAP_BY_SIZE_FIELD_NUMBER: builtins.int
    LOGICAL_BUFFER_SPANS_FIELD_NUMBER: builtins.int
    MAX_HEAP_TO_BY_SIZE_FIELD_NUMBER: builtins.int
    BY_SIZE_TO_MAX_HEAP_FIELD_NUMBER: builtins.int
    MODULE_NAME_FIELD_NUMBER: builtins.int
    ENTRY_COMPUTATION_NAME_FIELD_NUMBER: builtins.int
    PEAK_HEAP_MIB_FIELD_NUMBER: builtins.int
    PEAK_UNPADDED_HEAP_MIB_FIELD_NUMBER: builtins.int
    PEAK_HEAP_SIZE_POSITION_FIELD_NUMBER: builtins.int
    ENTRY_COMPUTATION_PARAMETERS_MIB_FIELD_NUMBER: builtins.int
    NON_REUSABLE_MIB_FIELD_NUMBER: builtins.int
    MAYBE_LIVE_OUT_MIB_FIELD_NUMBER: builtins.int
    INDEFINITE_LIFETIMES_FIELD_NUMBER: builtins.int
    @property
    def heap_sizes(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """Heap sizes at each HLO program point (the HLO sequential order)."""
    @property
    def unpadded_heap_sizes(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """Unpadded heap sizes (calculated as the minimal sizes based on the data type
        and dimensionality) at each HLO program point (the HLO sequential order).
        """
    @property
    def max_heap(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___HeapObject]:
        """Heap objects at the peak memory usage point ordered by HLO program "birth"
        time.
        """
    @property
    def max_heap_by_size(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___HeapObject]:
        """Heap objects at the peak memory usage point ordered by size, descending."""
    @property
    def logical_buffer_spans(self) -> google.protobuf.internal.containers.MessageMap[builtins.int, global___BufferSpan]:
        """Mapping from logical buffer ID to the HLO sequential order span in which it
        is alive.
        """
    @property
    def max_heap_to_by_size(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Indexes to get back and forth from the by-size and by-program-order
        sequences.
        """
    @property
    def by_size_to_max_heap(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    module_name: builtins.str
    entry_computation_name: builtins.str
    peak_heap_mib: builtins.float
    """Peak heap size for the HLO program."""
    peak_unpadded_heap_mib: builtins.float
    """Peak unpadded heap size for the HLO program."""
    peak_heap_size_position: builtins.int
    """HLO program point number at which the peak heap size occurs."""
    entry_computation_parameters_mib: builtins.float
    """Size of the entry computation parameters in MiB.

    This does not reflect whether those MiB are reusable during the computation
    or not, it is simply a size value.
    """
    non_reusable_mib: builtins.float
    maybe_live_out_mib: builtins.float
    @property
    def indefinite_lifetimes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___BufferAllocation]: ...
    def __init__(
        self,
        *,
        heap_sizes: collections.abc.Iterable[builtins.float] | None = ...,
        unpadded_heap_sizes: collections.abc.Iterable[builtins.float] | None = ...,
        max_heap: collections.abc.Iterable[global___HeapObject] | None = ...,
        max_heap_by_size: collections.abc.Iterable[global___HeapObject] | None = ...,
        logical_buffer_spans: collections.abc.Mapping[builtins.int, global___BufferSpan] | None = ...,
        max_heap_to_by_size: collections.abc.Iterable[builtins.int] | None = ...,
        by_size_to_max_heap: collections.abc.Iterable[builtins.int] | None = ...,
        module_name: builtins.str = ...,
        entry_computation_name: builtins.str = ...,
        peak_heap_mib: builtins.float = ...,
        peak_unpadded_heap_mib: builtins.float = ...,
        peak_heap_size_position: builtins.int = ...,
        entry_computation_parameters_mib: builtins.float = ...,
        non_reusable_mib: builtins.float = ...,
        maybe_live_out_mib: builtins.float = ...,
        indefinite_lifetimes: collections.abc.Iterable[global___BufferAllocation] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["by_size_to_max_heap", b"by_size_to_max_heap", "entry_computation_name", b"entry_computation_name", "entry_computation_parameters_mib", b"entry_computation_parameters_mib", "heap_sizes", b"heap_sizes", "indefinite_lifetimes", b"indefinite_lifetimes", "logical_buffer_spans", b"logical_buffer_spans", "max_heap", b"max_heap", "max_heap_by_size", b"max_heap_by_size", "max_heap_to_by_size", b"max_heap_to_by_size", "maybe_live_out_mib", b"maybe_live_out_mib", "module_name", b"module_name", "non_reusable_mib", b"non_reusable_mib", "peak_heap_mib", b"peak_heap_mib", "peak_heap_size_position", b"peak_heap_size_position", "peak_unpadded_heap_mib", b"peak_unpadded_heap_mib", "unpadded_heap_sizes", b"unpadded_heap_sizes"]) -> None: ...

global___PreprocessResult = PreprocessResult
