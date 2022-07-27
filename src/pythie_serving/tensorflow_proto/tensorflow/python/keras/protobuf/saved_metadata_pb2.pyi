"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow.python.keras.protobuf.versions_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class SavedMetadata(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NODES_FIELD_NUMBER: builtins.int
    @property
    def nodes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___SavedObject]:
        """Nodes represent trackable objects in the SavedModel. The data for every
        Keras object is stored.
        """
        pass
    def __init__(self,
        *,
        nodes: typing.Optional[typing.Iterable[global___SavedObject]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["nodes",b"nodes"]) -> None: ...
global___SavedMetadata = SavedMetadata

class SavedObject(google.protobuf.message.Message):
    """Metadata of an individual Keras object."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NODE_ID_FIELD_NUMBER: builtins.int
    NODE_PATH_FIELD_NUMBER: builtins.int
    IDENTIFIER_FIELD_NUMBER: builtins.int
    METADATA_FIELD_NUMBER: builtins.int
    VERSION_FIELD_NUMBER: builtins.int
    node_id: builtins.int
    """Index of the node in the SavedModel SavedObjectGraph."""

    node_path: typing.Text
    """String path from root (e.g. "root.child_layer")"""

    identifier: typing.Text
    """Identifier to determine loading function.
    Must be one of:
      _tf_keras_input_layer, _tf_keras_layer, _tf_keras_metric,
      _tf_keras_model, _tf_keras_network, _tf_keras_rnn_layer,
      _tf_keras_sequential
    """

    metadata: typing.Text
    """Metadata containing a JSON-serialized object with the non-TensorFlow
    attributes for this Keras object.
    """

    @property
    def version(self) -> tensorflow.python.keras.protobuf.versions_pb2.VersionDef:
        """Version defined by the code serializing this Keras object."""
        pass
    def __init__(self,
        *,
        node_id: builtins.int = ...,
        node_path: typing.Text = ...,
        identifier: typing.Text = ...,
        metadata: typing.Text = ...,
        version: typing.Optional[tensorflow.python.keras.protobuf.versions_pb2.VersionDef] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["version",b"version"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["identifier",b"identifier","metadata",b"metadata","node_id",b"node_id","node_path",b"node_path","version",b"version"]) -> None: ...
global___SavedObject = SavedObject