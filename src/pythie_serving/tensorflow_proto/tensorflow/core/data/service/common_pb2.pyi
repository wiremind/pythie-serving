"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import tensorflow.core.framework.graph_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _ProcessingModeDef:
    ValueType = typing.NewType('ValueType', builtins.int)
    V: typing_extensions.TypeAlias = ValueType
class _ProcessingModeDefEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_ProcessingModeDef.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    INVALID: _ProcessingModeDef.ValueType  # 0
    PARALLEL_EPOCHS: _ProcessingModeDef.ValueType  # 1
    """Each tf.data worker processes an entire epoch."""

    DISTRIBUTED_EPOCH: _ProcessingModeDef.ValueType  # 2
    """Processing of an epoch is distributed across all tf.data workers."""

class ProcessingModeDef(_ProcessingModeDef, metaclass=_ProcessingModeDefEnumTypeWrapper):
    """Next tag: 3"""
    pass

INVALID: ProcessingModeDef.ValueType  # 0
PARALLEL_EPOCHS: ProcessingModeDef.ValueType  # 1
"""Each tf.data worker processes an entire epoch."""

DISTRIBUTED_EPOCH: ProcessingModeDef.ValueType  # 2
"""Processing of an epoch is distributed across all tf.data workers."""

global___ProcessingModeDef = ProcessingModeDef


class DatasetDef(google.protobuf.message.Message):
    """Next tag: 2"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    GRAPH_FIELD_NUMBER: builtins.int
    @property
    def graph(self) -> tensorflow.core.framework.graph_pb2.GraphDef:
        """We represent datasets as tensorflow GraphDefs which define the operations
        needed to create a tf.data dataset.
        """
        pass
    def __init__(self,
        *,
        graph: typing.Optional[tensorflow.core.framework.graph_pb2.GraphDef] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["graph",b"graph"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["graph",b"graph"]) -> None: ...
global___DatasetDef = DatasetDef

class TaskDef(google.protobuf.message.Message):
    """Next tag: 10"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    DATASET_DEF_FIELD_NUMBER: builtins.int
    PATH_FIELD_NUMBER: builtins.int
    DATASET_ID_FIELD_NUMBER: builtins.int
    TASK_ID_FIELD_NUMBER: builtins.int
    JOB_ID_FIELD_NUMBER: builtins.int
    NUM_SPLIT_PROVIDERS_FIELD_NUMBER: builtins.int
    WORKER_ADDRESS_FIELD_NUMBER: builtins.int
    PROCESSING_MODE_FIELD_NUMBER: builtins.int
    NUM_CONSUMERS_FIELD_NUMBER: builtins.int
    @property
    def dataset_def(self) -> global___DatasetDef: ...
    path: typing.Text
    dataset_id: builtins.int
    task_id: builtins.int
    job_id: builtins.int
    num_split_providers: builtins.int
    """In distributed epoch processing mode, we use one split provider for each
    source that feeds into the dataset. In parallel_epochs mode,
    `num_split_providers` is always zero.
    """

    worker_address: typing.Text
    """Address of the worker that the task is assigned to."""

    processing_mode: global___ProcessingModeDef.ValueType
    num_consumers: builtins.int
    def __init__(self,
        *,
        dataset_def: typing.Optional[global___DatasetDef] = ...,
        path: typing.Text = ...,
        dataset_id: builtins.int = ...,
        task_id: builtins.int = ...,
        job_id: builtins.int = ...,
        num_split_providers: builtins.int = ...,
        worker_address: typing.Text = ...,
        processing_mode: global___ProcessingModeDef.ValueType = ...,
        num_consumers: builtins.int = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["dataset",b"dataset","dataset_def",b"dataset_def","num_consumers",b"num_consumers","optional_num_consumers",b"optional_num_consumers","path",b"path"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["dataset",b"dataset","dataset_def",b"dataset_def","dataset_id",b"dataset_id","job_id",b"job_id","num_consumers",b"num_consumers","num_split_providers",b"num_split_providers","optional_num_consumers",b"optional_num_consumers","path",b"path","processing_mode",b"processing_mode","task_id",b"task_id","worker_address",b"worker_address"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["dataset",b"dataset"]) -> typing.Optional[typing_extensions.Literal["dataset_def","path"]]: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["optional_num_consumers",b"optional_num_consumers"]) -> typing.Optional[typing_extensions.Literal["num_consumers"]]: ...
global___TaskDef = TaskDef

class TaskInfo(google.protobuf.message.Message):
    """Next tag: 6"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKER_ADDRESS_FIELD_NUMBER: builtins.int
    TRANSFER_ADDRESS_FIELD_NUMBER: builtins.int
    TASK_ID_FIELD_NUMBER: builtins.int
    JOB_ID_FIELD_NUMBER: builtins.int
    STARTING_ROUND_FIELD_NUMBER: builtins.int
    worker_address: typing.Text
    """The address of the worker processing the task."""

    transfer_address: typing.Text
    """The transfer address of the worker processing the task."""

    task_id: builtins.int
    """The task id."""

    job_id: builtins.int
    """The id of the job that the task is part of."""

    starting_round: builtins.int
    """The round to start reading from the task in. For non-round-robin reads,
    this is always 0.
    """

    def __init__(self,
        *,
        worker_address: typing.Text = ...,
        transfer_address: typing.Text = ...,
        task_id: builtins.int = ...,
        job_id: builtins.int = ...,
        starting_round: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["job_id",b"job_id","starting_round",b"starting_round","task_id",b"task_id","transfer_address",b"transfer_address","worker_address",b"worker_address"]) -> None: ...
global___TaskInfo = TaskInfo