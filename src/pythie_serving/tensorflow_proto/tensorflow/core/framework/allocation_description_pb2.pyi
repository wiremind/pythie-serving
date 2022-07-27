"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class AllocationDescription(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    REQUESTED_BYTES_FIELD_NUMBER: builtins.int
    ALLOCATED_BYTES_FIELD_NUMBER: builtins.int
    ALLOCATOR_NAME_FIELD_NUMBER: builtins.int
    ALLOCATION_ID_FIELD_NUMBER: builtins.int
    HAS_SINGLE_REFERENCE_FIELD_NUMBER: builtins.int
    PTR_FIELD_NUMBER: builtins.int
    requested_bytes: builtins.int
    """Total number of bytes requested"""

    allocated_bytes: builtins.int
    """Total number of bytes allocated if known"""

    allocator_name: typing.Text
    """Name of the allocator used"""

    allocation_id: builtins.int
    """Identifier of the allocated buffer if known"""

    has_single_reference: builtins.bool
    """Set if this tensor only has one remaining reference"""

    ptr: builtins.int
    """Address of the allocation."""

    def __init__(self,
        *,
        requested_bytes: builtins.int = ...,
        allocated_bytes: builtins.int = ...,
        allocator_name: typing.Text = ...,
        allocation_id: builtins.int = ...,
        has_single_reference: builtins.bool = ...,
        ptr: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["allocated_bytes",b"allocated_bytes","allocation_id",b"allocation_id","allocator_name",b"allocator_name","has_single_reference",b"has_single_reference","ptr",b"ptr","requested_bytes",b"requested_bytes"]) -> None: ...
global___AllocationDescription = AllocationDescription