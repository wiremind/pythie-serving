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
class ReaderBaseState(google.protobuf.message.Message):
    """For serializing and restoring the state of ReaderBase, see
    reader_base.h for details.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    WORK_STARTED_FIELD_NUMBER: builtins.int
    WORK_FINISHED_FIELD_NUMBER: builtins.int
    NUM_RECORDS_PRODUCED_FIELD_NUMBER: builtins.int
    CURRENT_WORK_FIELD_NUMBER: builtins.int
    work_started: builtins.int
    work_finished: builtins.int
    num_records_produced: builtins.int
    current_work: builtins.bytes
    def __init__(
        self,
        *,
        work_started: builtins.int = ...,
        work_finished: builtins.int = ...,
        num_records_produced: builtins.int = ...,
        current_work: builtins.bytes = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["current_work", b"current_work", "num_records_produced", b"num_records_produced", "work_finished", b"work_finished", "work_started", b"work_started"]) -> None: ...

global___ReaderBaseState = ReaderBaseState
