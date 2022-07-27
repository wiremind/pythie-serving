"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow.core.framework.tensor_shape_pb2
import tensorflow.core.framework.types_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class ResourceHandleProto(google.protobuf.message.Message):
    """Protocol buffer representing a handle to a tensorflow resource. Handles are
    not valid across executions, but can be serialized back and forth from within
    a single run.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class DtypeAndShape(google.protobuf.message.Message):
        """Protocol buffer representing a pair of (data type, tensor shape)."""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        DTYPE_FIELD_NUMBER: builtins.int
        SHAPE_FIELD_NUMBER: builtins.int
        dtype: tensorflow.core.framework.types_pb2.DataType.ValueType
        @property
        def shape(self) -> tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto: ...
        def __init__(self,
            *,
            dtype: tensorflow.core.framework.types_pb2.DataType.ValueType = ...,
            shape: typing.Optional[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["shape",b"shape"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["dtype",b"dtype","shape",b"shape"]) -> None: ...

    DEVICE_FIELD_NUMBER: builtins.int
    CONTAINER_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    HASH_CODE_FIELD_NUMBER: builtins.int
    MAYBE_TYPE_NAME_FIELD_NUMBER: builtins.int
    DTYPES_AND_SHAPES_FIELD_NUMBER: builtins.int
    device: typing.Text
    """Unique name for the device containing the resource."""

    container: typing.Text
    """Container in which this resource is placed."""

    name: typing.Text
    """Unique name of this resource."""

    hash_code: builtins.int
    """Hash code for the type of the resource. Is only valid in the same device
    and in the same execution.
    """

    maybe_type_name: typing.Text
    """For debug-only, the name of the type pointed to by this handle, if
    available.
    """

    @property
    def dtypes_and_shapes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ResourceHandleProto.DtypeAndShape]:
        """Data types and shapes for the underlying resource."""
        pass
    def __init__(self,
        *,
        device: typing.Text = ...,
        container: typing.Text = ...,
        name: typing.Text = ...,
        hash_code: builtins.int = ...,
        maybe_type_name: typing.Text = ...,
        dtypes_and_shapes: typing.Optional[typing.Iterable[global___ResourceHandleProto.DtypeAndShape]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["container",b"container","device",b"device","dtypes_and_shapes",b"dtypes_and_shapes","hash_code",b"hash_code","maybe_type_name",b"maybe_type_name","name",b"name"]) -> None: ...
global___ResourceHandleProto = ResourceHandleProto
