"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys
import tensorflow.core.data.dataset_pb2
import tensorflow.core.data.service.common_pb2
import typing

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class ProcessTaskRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TASK_FIELD_NUMBER: builtins.int
    @property
    def task(self) -> tensorflow.core.data.service.common_pb2.TaskDef: ...
    def __init__(
        self,
        *,
        task: tensorflow.core.data.service.common_pb2.TaskDef | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["task", b"task"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["task", b"task"]) -> None: ...

global___ProcessTaskRequest = ProcessTaskRequest

@typing_extensions.final
class ProcessTaskResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___ProcessTaskResponse = ProcessTaskResponse

@typing_extensions.final
class GetElementRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TASK_ID_FIELD_NUMBER: builtins.int
    CONSUMER_INDEX_FIELD_NUMBER: builtins.int
    ROUND_INDEX_FIELD_NUMBER: builtins.int
    SKIPPED_PREVIOUS_ROUND_FIELD_NUMBER: builtins.int
    ALLOW_SKIP_FIELD_NUMBER: builtins.int
    TRAINER_ID_FIELD_NUMBER: builtins.int
    task_id: builtins.int
    """The task to fetch an element from."""
    consumer_index: builtins.int
    round_index: builtins.int
    skipped_previous_round: builtins.bool
    """Whether the previous round was skipped. This information is needed by the
    worker to recover after restarts.
    """
    allow_skip: builtins.bool
    """Whether to skip the round if data isn't ready fast enough."""
    trainer_id: builtins.str
    """The trainer ID used to read elements from a multi-trainer cache. This cache
    enables sharing data across concurrent training iterations. If set, this
    request will read the data requested by other trainers, if available.
    """
    def __init__(
        self,
        *,
        task_id: builtins.int = ...,
        consumer_index: builtins.int = ...,
        round_index: builtins.int = ...,
        skipped_previous_round: builtins.bool = ...,
        allow_skip: builtins.bool = ...,
        trainer_id: builtins.str = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["consumer_index", b"consumer_index", "optional_consumer_index", b"optional_consumer_index", "optional_round_index", b"optional_round_index", "round_index", b"round_index"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["allow_skip", b"allow_skip", "consumer_index", b"consumer_index", "optional_consumer_index", b"optional_consumer_index", "optional_round_index", b"optional_round_index", "round_index", b"round_index", "skipped_previous_round", b"skipped_previous_round", "task_id", b"task_id", "trainer_id", b"trainer_id"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["optional_consumer_index", b"optional_consumer_index"]) -> typing_extensions.Literal["consumer_index"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["optional_round_index", b"optional_round_index"]) -> typing_extensions.Literal["round_index"] | None: ...

global___GetElementRequest = GetElementRequest

@typing_extensions.final
class GetElementResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    COMPRESSED_FIELD_NUMBER: builtins.int
    UNCOMPRESSED_FIELD_NUMBER: builtins.int
    ELEMENT_INDEX_FIELD_NUMBER: builtins.int
    END_OF_SEQUENCE_FIELD_NUMBER: builtins.int
    SKIP_TASK_FIELD_NUMBER: builtins.int
    @property
    def compressed(self) -> tensorflow.core.data.dataset_pb2.CompressedElement: ...
    @property
    def uncompressed(self) -> tensorflow.core.data.dataset_pb2.UncompressedElement: ...
    element_index: builtins.int
    """The element's index within the task it came from."""
    end_of_sequence: builtins.bool
    """Boolean to indicate whether the iterator has been exhausted."""
    skip_task: builtins.bool
    """Indicates whether the round was skipped."""
    def __init__(
        self,
        *,
        compressed: tensorflow.core.data.dataset_pb2.CompressedElement | None = ...,
        uncompressed: tensorflow.core.data.dataset_pb2.UncompressedElement | None = ...,
        element_index: builtins.int = ...,
        end_of_sequence: builtins.bool = ...,
        skip_task: builtins.bool = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["compressed", b"compressed", "element", b"element", "uncompressed", b"uncompressed"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["compressed", b"compressed", "element", b"element", "element_index", b"element_index", "end_of_sequence", b"end_of_sequence", "skip_task", b"skip_task", "uncompressed", b"uncompressed"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["element", b"element"]) -> typing_extensions.Literal["compressed", "uncompressed"] | None: ...

global___GetElementResponse = GetElementResponse

@typing_extensions.final
class GetWorkerTasksRequest(google.protobuf.message.Message):
    """Named GetWorkerTasks to avoid conflicting with GetTasks in dispatcher.proto"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___GetWorkerTasksRequest = GetWorkerTasksRequest

@typing_extensions.final
class GetWorkerTasksResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TASKS_FIELD_NUMBER: builtins.int
    @property
    def tasks(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tensorflow.core.data.service.common_pb2.TaskInfo]: ...
    def __init__(
        self,
        *,
        tasks: collections.abc.Iterable[tensorflow.core.data.service.common_pb2.TaskInfo] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["tasks", b"tasks"]) -> None: ...

global___GetWorkerTasksResponse = GetWorkerTasksResponse
