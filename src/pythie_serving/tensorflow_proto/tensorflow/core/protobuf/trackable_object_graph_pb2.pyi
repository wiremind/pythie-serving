"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class TrackableObjectGraph(google.protobuf.message.Message):
    """A TensorBundle addition which saves extra information about the objects which
    own variables, allowing for more robust checkpoint loading into modified
    programs.

    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class TrackableObject(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        class ObjectReference(google.protobuf.message.Message):
            DESCRIPTOR: google.protobuf.descriptor.Descriptor
            NODE_ID_FIELD_NUMBER: builtins.int
            LOCAL_NAME_FIELD_NUMBER: builtins.int
            node_id: builtins.int
            """An index into `TrackableObjectGraph.nodes`, indicating the object
            being referenced.
            """

            local_name: typing.Text
            """A user-provided name for the edge."""

            def __init__(self,
                *,
                node_id: builtins.int = ...,
                local_name: typing.Text = ...,
                ) -> None: ...
            def ClearField(self, field_name: typing_extensions.Literal["local_name",b"local_name","node_id",b"node_id"]) -> None: ...

        class SerializedTensor(google.protobuf.message.Message):
            DESCRIPTOR: google.protobuf.descriptor.Descriptor
            NAME_FIELD_NUMBER: builtins.int
            FULL_NAME_FIELD_NUMBER: builtins.int
            CHECKPOINT_KEY_FIELD_NUMBER: builtins.int
            OPTIONAL_RESTORE_FIELD_NUMBER: builtins.int
            name: typing.Text
            """A name for the Tensor. Simple variables have only one
            `SerializedTensor` named "VARIABLE_VALUE" by convention. This value may
            be restored on object creation as an optimization.
            """

            full_name: typing.Text
            """The full name of the variable/tensor, if applicable. Used to allow
            name-based loading of checkpoints which were saved using an
            object-based API. Should match the checkpoint key which would have been
            assigned by tf.train.Saver.
            """

            checkpoint_key: typing.Text
            """The generated name of the Tensor in the checkpoint."""

            optional_restore: builtins.bool
            """Whether checkpoints should be considered as matching even without this
            value restored. Used for non-critical values which don't affect the
            TensorFlow graph, such as layer configurations.
            """

            def __init__(self,
                *,
                name: typing.Text = ...,
                full_name: typing.Text = ...,
                checkpoint_key: typing.Text = ...,
                optional_restore: builtins.bool = ...,
                ) -> None: ...
            def ClearField(self, field_name: typing_extensions.Literal["checkpoint_key",b"checkpoint_key","full_name",b"full_name","name",b"name","optional_restore",b"optional_restore"]) -> None: ...

        class SlotVariableReference(google.protobuf.message.Message):
            DESCRIPTOR: google.protobuf.descriptor.Descriptor
            ORIGINAL_VARIABLE_NODE_ID_FIELD_NUMBER: builtins.int
            SLOT_NAME_FIELD_NUMBER: builtins.int
            SLOT_VARIABLE_NODE_ID_FIELD_NUMBER: builtins.int
            original_variable_node_id: builtins.int
            """An index into `TrackableObjectGraph.nodes`, indicating the
            variable object this slot was created for.
            """

            slot_name: typing.Text
            """The name of the slot (e.g. "m"/"v")."""

            slot_variable_node_id: builtins.int
            """An index into `TrackableObjectGraph.nodes`, indicating the
            `Object` with the value of the slot variable.
            """

            def __init__(self,
                *,
                original_variable_node_id: builtins.int = ...,
                slot_name: typing.Text = ...,
                slot_variable_node_id: builtins.int = ...,
                ) -> None: ...
            def ClearField(self, field_name: typing_extensions.Literal["original_variable_node_id",b"original_variable_node_id","slot_name",b"slot_name","slot_variable_node_id",b"slot_variable_node_id"]) -> None: ...

        CHILDREN_FIELD_NUMBER: builtins.int
        ATTRIBUTES_FIELD_NUMBER: builtins.int
        SLOT_VARIABLES_FIELD_NUMBER: builtins.int
        @property
        def children(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TrackableObjectGraph.TrackableObject.ObjectReference]:
            """Objects which this object depends on."""
            pass
        @property
        def attributes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TrackableObjectGraph.TrackableObject.SerializedTensor]:
            """Serialized data specific to this object."""
            pass
        @property
        def slot_variables(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TrackableObjectGraph.TrackableObject.SlotVariableReference]:
            """Slot variables owned by this object."""
            pass
        def __init__(self,
            *,
            children: typing.Optional[typing.Iterable[global___TrackableObjectGraph.TrackableObject.ObjectReference]] = ...,
            attributes: typing.Optional[typing.Iterable[global___TrackableObjectGraph.TrackableObject.SerializedTensor]] = ...,
            slot_variables: typing.Optional[typing.Iterable[global___TrackableObjectGraph.TrackableObject.SlotVariableReference]] = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["attributes",b"attributes","children",b"children","slot_variables",b"slot_variables"]) -> None: ...

    NODES_FIELD_NUMBER: builtins.int
    @property
    def nodes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___TrackableObjectGraph.TrackableObject]: ...
    def __init__(self,
        *,
        nodes: typing.Optional[typing.Iterable[global___TrackableObjectGraph.TrackableObject]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["nodes",b"nodes"]) -> None: ...
global___TrackableObjectGraph = TrackableObjectGraph