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
import tensorflow.core.framework.tensor_shape_pb2

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class TRTEngineInstance(google.protobuf.message.Message):
    """Containing information for a serialized TensorRT engine."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INPUT_SHAPES_FIELD_NUMBER: builtins.int
    SERIALIZED_ENGINE_FIELD_NUMBER: builtins.int
    @property
    def input_shapes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto]:
        """The input shapes of the TRT engine."""
    serialized_engine: builtins.bytes
    """The serialized TRT engine.

    TODO(laigd): consider using a more efficient in-memory representation
    instead of string which is the default here.
    """
    def __init__(
        self,
        *,
        input_shapes: collections.abc.Iterable[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto] | None = ...,
        serialized_engine: builtins.bytes = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["input_shapes", b"input_shapes", "serialized_engine", b"serialized_engine"]) -> None: ...

global___TRTEngineInstance = TRTEngineInstance
