"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import tensorflow_serving.apis.model_pb2
import tensorflow_serving.apis.status_pb2
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class GetModelStatusRequest(google.protobuf.message.Message):
    """GetModelStatusRequest contains a ModelSpec indicating the model for which
    to get status.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_SPEC_FIELD_NUMBER: builtins.int
    @property
    def model_spec(self) -> tensorflow_serving.apis.model_pb2.ModelSpec:
        """Model Specification. If version is not specified, information about all
        versions of the model will be returned. If a version is specified, the
        status of only that version will be returned.
        """
    def __init__(
        self,
        *,
        model_spec: tensorflow_serving.apis.model_pb2.ModelSpec | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["model_spec", b"model_spec"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["model_spec", b"model_spec"]) -> None: ...

global___GetModelStatusRequest = GetModelStatusRequest

@typing_extensions.final
class ModelVersionStatus(google.protobuf.message.Message):
    """Version number, state, and status for a single version of a model."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _State:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _StateEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[ModelVersionStatus._State.ValueType], builtins.type):  # noqa: F821
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        UNKNOWN: ModelVersionStatus._State.ValueType  # 0
        """Default value."""
        START: ModelVersionStatus._State.ValueType  # 10
        """The manager is tracking this servable, but has not initiated any action
        pertaining to it.
        """
        LOADING: ModelVersionStatus._State.ValueType  # 20
        """The manager has decided to load this servable. In particular, checks
        around resource availability and other aspects have passed, and the
        manager is about to invoke the loader's Load() method.
        """
        AVAILABLE: ModelVersionStatus._State.ValueType  # 30
        """The manager has successfully loaded this servable and made it available
        for serving (i.e. GetServableHandle(id) will succeed). To avoid races,
        this state is not reported until *after* the servable is made
        available.
        """
        UNLOADING: ModelVersionStatus._State.ValueType  # 40
        """The manager has decided to make this servable unavailable, and unload
        it. To avoid races, this state is reported *before* the servable is
        made unavailable.
        """
        END: ModelVersionStatus._State.ValueType  # 50
        """This servable has reached the end of its journey in the manager. Either
        it loaded and ultimately unloaded successfully, or it hit an error at
        some point in its lifecycle.
        """

    class State(_State, metaclass=_StateEnumTypeWrapper):
        """States that map to ManagerState enum in
        tensorflow_serving/core/servable_state.h
        """

    UNKNOWN: ModelVersionStatus.State.ValueType  # 0
    """Default value."""
    START: ModelVersionStatus.State.ValueType  # 10
    """The manager is tracking this servable, but has not initiated any action
    pertaining to it.
    """
    LOADING: ModelVersionStatus.State.ValueType  # 20
    """The manager has decided to load this servable. In particular, checks
    around resource availability and other aspects have passed, and the
    manager is about to invoke the loader's Load() method.
    """
    AVAILABLE: ModelVersionStatus.State.ValueType  # 30
    """The manager has successfully loaded this servable and made it available
    for serving (i.e. GetServableHandle(id) will succeed). To avoid races,
    this state is not reported until *after* the servable is made
    available.
    """
    UNLOADING: ModelVersionStatus.State.ValueType  # 40
    """The manager has decided to make this servable unavailable, and unload
    it. To avoid races, this state is reported *before* the servable is
    made unavailable.
    """
    END: ModelVersionStatus.State.ValueType  # 50
    """This servable has reached the end of its journey in the manager. Either
    it loaded and ultimately unloaded successfully, or it hit an error at
    some point in its lifecycle.
    """

    VERSION_FIELD_NUMBER: builtins.int
    STATE_FIELD_NUMBER: builtins.int
    STATUS_FIELD_NUMBER: builtins.int
    version: builtins.int
    """Model version."""
    state: global___ModelVersionStatus.State.ValueType
    """Model state."""
    @property
    def status(self) -> tensorflow_serving.apis.status_pb2.StatusProto:
        """Model status."""
    def __init__(
        self,
        *,
        version: builtins.int = ...,
        state: global___ModelVersionStatus.State.ValueType = ...,
        status: tensorflow_serving.apis.status_pb2.StatusProto | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["status", b"status"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["state", b"state", "status", b"status", "version", b"version"]) -> None: ...

global___ModelVersionStatus = ModelVersionStatus

@typing_extensions.final
class GetModelStatusResponse(google.protobuf.message.Message):
    """Response for ModelStatusRequest on successful run."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_VERSION_STATUS_FIELD_NUMBER: builtins.int
    @property
    def model_version_status(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ModelVersionStatus]:
        """Version number and status information for applicable model version(s)."""
    def __init__(
        self,
        *,
        model_version_status: collections.abc.Iterable[global___ModelVersionStatus] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["model_version_status", b"model_version_status"]) -> None: ...

global___GetModelStatusResponse = GetModelStatusResponse
