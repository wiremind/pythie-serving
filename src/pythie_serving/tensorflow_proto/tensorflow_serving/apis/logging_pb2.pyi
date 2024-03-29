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
import tensorflow_serving.apis.model_pb2
import tensorflow_serving.config.logging_config_pb2

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class LogMetadata(google.protobuf.message.Message):
    """Metadata logged along with the request logs."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_SPEC_FIELD_NUMBER: builtins.int
    SAMPLING_CONFIG_FIELD_NUMBER: builtins.int
    SAVED_MODEL_TAGS_FIELD_NUMBER: builtins.int
    @property
    def model_spec(self) -> tensorflow_serving.apis.model_pb2.ModelSpec: ...
    @property
    def sampling_config(self) -> tensorflow_serving.config.logging_config_pb2.SamplingConfig: ...
    @property
    def saved_model_tags(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """List of tags used to load the relevant MetaGraphDef from SavedModel.
        TODO(b/33279154): Add more metadata as mentioned in the bug.
        """
    def __init__(
        self,
        *,
        model_spec: tensorflow_serving.apis.model_pb2.ModelSpec | None = ...,
        sampling_config: tensorflow_serving.config.logging_config_pb2.SamplingConfig | None = ...,
        saved_model_tags: collections.abc.Iterable[builtins.str] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["model_spec", b"model_spec", "sampling_config", b"sampling_config"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["model_spec", b"model_spec", "sampling_config", b"sampling_config", "saved_model_tags", b"saved_model_tags"]) -> None: ...

global___LogMetadata = LogMetadata
