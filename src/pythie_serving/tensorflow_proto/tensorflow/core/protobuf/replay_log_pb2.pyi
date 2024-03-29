"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.message
import sys
import tensorflow.core.protobuf.master_pb2
import typing

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class NewReplaySession(google.protobuf.message.Message):
    """Records the creation of a new replay session.  We record the device listing
    here to capture the state of the cluster.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DEVICES_FIELD_NUMBER: builtins.int
    SESSION_HANDLE_FIELD_NUMBER: builtins.int
    @property
    def devices(self) -> tensorflow.core.protobuf.master_pb2.ListDevicesResponse: ...
    session_handle: builtins.str
    def __init__(
        self,
        *,
        devices: tensorflow.core.protobuf.master_pb2.ListDevicesResponse | None = ...,
        session_handle: builtins.str = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["devices", b"devices"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["devices", b"devices", "session_handle", b"session_handle"]) -> None: ...

global___NewReplaySession = NewReplaySession

@typing_extensions.final
class ReplayOp(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    START_TIME_US_FIELD_NUMBER: builtins.int
    END_TIME_US_FIELD_NUMBER: builtins.int
    CREATE_SESSION_FIELD_NUMBER: builtins.int
    EXTEND_SESSION_FIELD_NUMBER: builtins.int
    PARTIAL_RUN_SETUP_FIELD_NUMBER: builtins.int
    RUN_STEP_FIELD_NUMBER: builtins.int
    CLOSE_SESSION_FIELD_NUMBER: builtins.int
    LIST_DEVICES_FIELD_NUMBER: builtins.int
    RESET_REQUEST_FIELD_NUMBER: builtins.int
    MAKE_CALLABLE_FIELD_NUMBER: builtins.int
    RUN_CALLABLE_FIELD_NUMBER: builtins.int
    RELEASE_CALLABLE_FIELD_NUMBER: builtins.int
    NEW_REPLAY_SESSION_FIELD_NUMBER: builtins.int
    CREATE_SESSION_RESPONSE_FIELD_NUMBER: builtins.int
    EXTEND_SESSION_RESPONSE_FIELD_NUMBER: builtins.int
    PARTIAL_RUN_SETUP_RESPONSE_FIELD_NUMBER: builtins.int
    RUN_STEP_RESPONSE_FIELD_NUMBER: builtins.int
    CLOSE_SESSION_RESPONSE_FIELD_NUMBER: builtins.int
    LIST_DEVICES_RESPONSE_FIELD_NUMBER: builtins.int
    RESET_REQUEST_RESPONSE_FIELD_NUMBER: builtins.int
    MAKE_CALLABLE_RESPONSE_FIELD_NUMBER: builtins.int
    RUN_CALLABLE_RESPONSE_FIELD_NUMBER: builtins.int
    RELEASE_CALLABLE_RESPONSE_FIELD_NUMBER: builtins.int
    start_time_us: builtins.float
    end_time_us: builtins.float
    @property
    def create_session(self) -> tensorflow.core.protobuf.master_pb2.CreateSessionRequest: ...
    @property
    def extend_session(self) -> tensorflow.core.protobuf.master_pb2.ExtendSessionRequest: ...
    @property
    def partial_run_setup(self) -> tensorflow.core.protobuf.master_pb2.PartialRunSetupRequest: ...
    @property
    def run_step(self) -> tensorflow.core.protobuf.master_pb2.RunStepRequest: ...
    @property
    def close_session(self) -> tensorflow.core.protobuf.master_pb2.CloseSessionRequest: ...
    @property
    def list_devices(self) -> tensorflow.core.protobuf.master_pb2.ListDevicesRequest: ...
    @property
    def reset_request(self) -> tensorflow.core.protobuf.master_pb2.ResetRequest: ...
    @property
    def make_callable(self) -> tensorflow.core.protobuf.master_pb2.MakeCallableRequest: ...
    @property
    def run_callable(self) -> tensorflow.core.protobuf.master_pb2.RunCallableRequest: ...
    @property
    def release_callable(self) -> tensorflow.core.protobuf.master_pb2.ReleaseCallableRequest: ...
    @property
    def new_replay_session(self) -> global___NewReplaySession: ...
    @property
    def create_session_response(self) -> tensorflow.core.protobuf.master_pb2.CreateSessionResponse: ...
    @property
    def extend_session_response(self) -> tensorflow.core.protobuf.master_pb2.ExtendSessionResponse: ...
    @property
    def partial_run_setup_response(self) -> tensorflow.core.protobuf.master_pb2.PartialRunSetupResponse: ...
    @property
    def run_step_response(self) -> tensorflow.core.protobuf.master_pb2.RunStepResponse: ...
    @property
    def close_session_response(self) -> tensorflow.core.protobuf.master_pb2.CloseSessionResponse: ...
    @property
    def list_devices_response(self) -> tensorflow.core.protobuf.master_pb2.ListDevicesResponse: ...
    @property
    def reset_request_response(self) -> tensorflow.core.protobuf.master_pb2.ResetResponse: ...
    @property
    def make_callable_response(self) -> tensorflow.core.protobuf.master_pb2.MakeCallableResponse: ...
    @property
    def run_callable_response(self) -> tensorflow.core.protobuf.master_pb2.RunCallableResponse: ...
    @property
    def release_callable_response(self) -> tensorflow.core.protobuf.master_pb2.ReleaseCallableResponse: ...
    def __init__(
        self,
        *,
        start_time_us: builtins.float = ...,
        end_time_us: builtins.float = ...,
        create_session: tensorflow.core.protobuf.master_pb2.CreateSessionRequest | None = ...,
        extend_session: tensorflow.core.protobuf.master_pb2.ExtendSessionRequest | None = ...,
        partial_run_setup: tensorflow.core.protobuf.master_pb2.PartialRunSetupRequest | None = ...,
        run_step: tensorflow.core.protobuf.master_pb2.RunStepRequest | None = ...,
        close_session: tensorflow.core.protobuf.master_pb2.CloseSessionRequest | None = ...,
        list_devices: tensorflow.core.protobuf.master_pb2.ListDevicesRequest | None = ...,
        reset_request: tensorflow.core.protobuf.master_pb2.ResetRequest | None = ...,
        make_callable: tensorflow.core.protobuf.master_pb2.MakeCallableRequest | None = ...,
        run_callable: tensorflow.core.protobuf.master_pb2.RunCallableRequest | None = ...,
        release_callable: tensorflow.core.protobuf.master_pb2.ReleaseCallableRequest | None = ...,
        new_replay_session: global___NewReplaySession | None = ...,
        create_session_response: tensorflow.core.protobuf.master_pb2.CreateSessionResponse | None = ...,
        extend_session_response: tensorflow.core.protobuf.master_pb2.ExtendSessionResponse | None = ...,
        partial_run_setup_response: tensorflow.core.protobuf.master_pb2.PartialRunSetupResponse | None = ...,
        run_step_response: tensorflow.core.protobuf.master_pb2.RunStepResponse | None = ...,
        close_session_response: tensorflow.core.protobuf.master_pb2.CloseSessionResponse | None = ...,
        list_devices_response: tensorflow.core.protobuf.master_pb2.ListDevicesResponse | None = ...,
        reset_request_response: tensorflow.core.protobuf.master_pb2.ResetResponse | None = ...,
        make_callable_response: tensorflow.core.protobuf.master_pb2.MakeCallableResponse | None = ...,
        run_callable_response: tensorflow.core.protobuf.master_pb2.RunCallableResponse | None = ...,
        release_callable_response: tensorflow.core.protobuf.master_pb2.ReleaseCallableResponse | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["close_session", b"close_session", "close_session_response", b"close_session_response", "create_session", b"create_session", "create_session_response", b"create_session_response", "extend_session", b"extend_session", "extend_session_response", b"extend_session_response", "list_devices", b"list_devices", "list_devices_response", b"list_devices_response", "make_callable", b"make_callable", "make_callable_response", b"make_callable_response", "new_replay_session", b"new_replay_session", "op", b"op", "partial_run_setup", b"partial_run_setup", "partial_run_setup_response", b"partial_run_setup_response", "release_callable", b"release_callable", "release_callable_response", b"release_callable_response", "reset_request", b"reset_request", "reset_request_response", b"reset_request_response", "response", b"response", "run_callable", b"run_callable", "run_callable_response", b"run_callable_response", "run_step", b"run_step", "run_step_response", b"run_step_response"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["close_session", b"close_session", "close_session_response", b"close_session_response", "create_session", b"create_session", "create_session_response", b"create_session_response", "end_time_us", b"end_time_us", "extend_session", b"extend_session", "extend_session_response", b"extend_session_response", "list_devices", b"list_devices", "list_devices_response", b"list_devices_response", "make_callable", b"make_callable", "make_callable_response", b"make_callable_response", "new_replay_session", b"new_replay_session", "op", b"op", "partial_run_setup", b"partial_run_setup", "partial_run_setup_response", b"partial_run_setup_response", "release_callable", b"release_callable", "release_callable_response", b"release_callable_response", "reset_request", b"reset_request", "reset_request_response", b"reset_request_response", "response", b"response", "run_callable", b"run_callable", "run_callable_response", b"run_callable_response", "run_step", b"run_step", "run_step_response", b"run_step_response", "start_time_us", b"start_time_us"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["op", b"op"]) -> typing_extensions.Literal["create_session", "extend_session", "partial_run_setup", "run_step", "close_session", "list_devices", "reset_request", "make_callable", "run_callable", "release_callable", "new_replay_session"] | None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["response", b"response"]) -> typing_extensions.Literal["create_session_response", "extend_session_response", "partial_run_setup_response", "run_step_response", "close_session_response", "list_devices_response", "reset_request_response", "make_callable_response", "run_callable_response", "release_callable_response"] | None: ...

global___ReplayOp = ReplayOp
