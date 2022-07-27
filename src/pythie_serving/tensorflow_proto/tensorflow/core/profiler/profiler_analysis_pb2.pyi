"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow.core.profiler.profiler_service_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class NewProfileSessionRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    REQUEST_FIELD_NUMBER: builtins.int
    REPOSITORY_ROOT_FIELD_NUMBER: builtins.int
    HOSTS_FIELD_NUMBER: builtins.int
    SESSION_ID_FIELD_NUMBER: builtins.int
    @property
    def request(self) -> tensorflow.core.profiler.profiler_service_pb2.ProfileRequest: ...
    repository_root: typing.Text
    """The place where we will dump profile data. We will normally use
    MODEL_DIR/plugins/profile as the repository root.
    """

    @property
    def hosts(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """host or host:port, port will be ignored."""
        pass
    session_id: typing.Text
    def __init__(self,
        *,
        request: typing.Optional[tensorflow.core.profiler.profiler_service_pb2.ProfileRequest] = ...,
        repository_root: typing.Text = ...,
        hosts: typing.Optional[typing.Iterable[typing.Text]] = ...,
        session_id: typing.Text = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["request",b"request"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["hosts",b"hosts","repository_root",b"repository_root","request",b"request","session_id",b"session_id"]) -> None: ...
global___NewProfileSessionRequest = NewProfileSessionRequest

class NewProfileSessionResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ERROR_MESSAGE_FIELD_NUMBER: builtins.int
    EMPTY_TRACE_FIELD_NUMBER: builtins.int
    error_message: typing.Text
    """Auxiliary error_message."""

    empty_trace: builtins.bool
    """Whether all hosts had returned a empty trace."""

    def __init__(self,
        *,
        error_message: typing.Text = ...,
        empty_trace: builtins.bool = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["empty_trace",b"empty_trace","error_message",b"error_message"]) -> None: ...
global___NewProfileSessionResponse = NewProfileSessionResponse

class EnumProfileSessionsAndToolsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    REPOSITORY_ROOT_FIELD_NUMBER: builtins.int
    repository_root: typing.Text
    def __init__(self,
        *,
        repository_root: typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["repository_root",b"repository_root"]) -> None: ...
global___EnumProfileSessionsAndToolsRequest = EnumProfileSessionsAndToolsRequest

class ProfileSessionInfo(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    SESSION_ID_FIELD_NUMBER: builtins.int
    AVAILABLE_TOOLS_FIELD_NUMBER: builtins.int
    session_id: typing.Text
    @property
    def available_tools(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """Which tool data is available for consumption."""
        pass
    def __init__(self,
        *,
        session_id: typing.Text = ...,
        available_tools: typing.Optional[typing.Iterable[typing.Text]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["available_tools",b"available_tools","session_id",b"session_id"]) -> None: ...
global___ProfileSessionInfo = ProfileSessionInfo

class EnumProfileSessionsAndToolsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ERROR_MESSAGE_FIELD_NUMBER: builtins.int
    SESSIONS_FIELD_NUMBER: builtins.int
    error_message: typing.Text
    """Auxiliary error_message."""

    @property
    def sessions(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ProfileSessionInfo]:
        """If success, the returned sessions information are stored here."""
        pass
    def __init__(self,
        *,
        error_message: typing.Text = ...,
        sessions: typing.Optional[typing.Iterable[global___ProfileSessionInfo]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["error_message",b"error_message","sessions",b"sessions"]) -> None: ...
global___EnumProfileSessionsAndToolsResponse = EnumProfileSessionsAndToolsResponse

class ProfileSessionDataRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class ParametersEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        value: typing.Text
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Text = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    REPOSITORY_ROOT_FIELD_NUMBER: builtins.int
    SESSION_ID_FIELD_NUMBER: builtins.int
    HOST_NAME_FIELD_NUMBER: builtins.int
    TOOL_NAME_FIELD_NUMBER: builtins.int
    PARAMETERS_FIELD_NUMBER: builtins.int
    repository_root: typing.Text
    """The place where we will read profile data. We will normally use
    MODEL_DIR/plugins/profile as the repository root.
    """

    session_id: typing.Text
    host_name: typing.Text
    """Which host the data is associated. if empty, data from all hosts are
    aggregated.
    """

    tool_name: typing.Text
    """Which tool"""

    @property
    def parameters(self) -> google.protobuf.internal.containers.ScalarMap[typing.Text, typing.Text]:
        """Tool's specific parameters. e.g. TraceViewer's viewport etc"""
        pass
    def __init__(self,
        *,
        repository_root: typing.Text = ...,
        session_id: typing.Text = ...,
        host_name: typing.Text = ...,
        tool_name: typing.Text = ...,
        parameters: typing.Optional[typing.Mapping[typing.Text, typing.Text]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["host_name",b"host_name","parameters",b"parameters","repository_root",b"repository_root","session_id",b"session_id","tool_name",b"tool_name"]) -> None: ...
global___ProfileSessionDataRequest = ProfileSessionDataRequest

class ProfileSessionDataResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    ERROR_MESSAGE_FIELD_NUMBER: builtins.int
    OUTPUT_FORMAT_FIELD_NUMBER: builtins.int
    OUTPUT_FIELD_NUMBER: builtins.int
    error_message: typing.Text
    """Auxiliary error_message."""

    output_format: typing.Text
    """Output format. e.g. "json" or "proto" or "blob" """

    output: builtins.bytes
    """TODO(jiesun): figure out whether to put bytes or oneof tool specific proto."""

    def __init__(self,
        *,
        error_message: typing.Text = ...,
        output_format: typing.Text = ...,
        output: builtins.bytes = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["error_message",b"error_message","output",b"output","output_format",b"output_format"]) -> None: ...
global___ProfileSessionDataResponse = ProfileSessionDataResponse