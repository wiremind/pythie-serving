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

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class CheckpointState(google.protobuf.message.Message):
    """Protocol buffer representing the checkpoint state."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_CHECKPOINT_PATH_FIELD_NUMBER: builtins.int
    ALL_MODEL_CHECKPOINT_PATHS_FIELD_NUMBER: builtins.int
    ALL_MODEL_CHECKPOINT_TIMESTAMPS_FIELD_NUMBER: builtins.int
    LAST_PRESERVED_TIMESTAMP_FIELD_NUMBER: builtins.int
    model_checkpoint_path: builtins.str
    """Path to the most-recent model checkpoint."""
    @property
    def all_model_checkpoint_paths(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Paths to all not-yet-deleted model checkpoints, sorted from oldest to
        newest.
        Note that the value of model_checkpoint_path should be the last item in
        this list.
        """
    @property
    def all_model_checkpoint_timestamps(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """Unix timestamps corresponding to all_model_checkpoint_paths, indicating
        when each checkpoint was created.
        """
    last_preserved_timestamp: builtins.float
    """Unix timestamp indicating the creation time for the last preserved
    checkpoint.
    """
    def __init__(
        self,
        *,
        model_checkpoint_path: builtins.str = ...,
        all_model_checkpoint_paths: collections.abc.Iterable[builtins.str] | None = ...,
        all_model_checkpoint_timestamps: collections.abc.Iterable[builtins.float] | None = ...,
        last_preserved_timestamp: builtins.float = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["all_model_checkpoint_paths", b"all_model_checkpoint_paths", "all_model_checkpoint_timestamps", b"all_model_checkpoint_timestamps", "last_preserved_timestamp", b"last_preserved_timestamp", "model_checkpoint_path", b"model_checkpoint_path"]) -> None: ...

global___CheckpointState = CheckpointState
