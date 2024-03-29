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
import tensorflow.core.profiler.profiler_options_pb2
import tensorflow.core.profiler.profiler_service_monitor_result_pb2

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class ToolRequestOptions(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    OUTPUT_FORMATS_FIELD_NUMBER: builtins.int
    SAVE_TO_REPO_FIELD_NUMBER: builtins.int
    output_formats: builtins.str
    """Required formats for the tool, it should be one of "json", "proto", "raw"
    etc. If not specified (backward compatible), use default format, i.e. most
    tools use json format.
    """
    save_to_repo: builtins.bool
    """Whether save the result directly to repository or pass it back to caller.
    Default to false for backward compatibilities.
    """
    def __init__(
        self,
        *,
        output_formats: builtins.str = ...,
        save_to_repo: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["output_formats", b"output_formats", "save_to_repo", b"save_to_repo"]) -> None: ...

global___ToolRequestOptions = ToolRequestOptions

@typing_extensions.final
class ProfileRequest(google.protobuf.message.Message):
    """Next-ID: 9"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class ToolOptionsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        @property
        def value(self) -> global___ToolRequestOptions: ...
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: global___ToolRequestOptions | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    DURATION_MS_FIELD_NUMBER: builtins.int
    MAX_EVENTS_FIELD_NUMBER: builtins.int
    TOOLS_FIELD_NUMBER: builtins.int
    TOOL_OPTIONS_FIELD_NUMBER: builtins.int
    OPTS_FIELD_NUMBER: builtins.int
    REPOSITORY_ROOT_FIELD_NUMBER: builtins.int
    SESSION_ID_FIELD_NUMBER: builtins.int
    HOST_NAME_FIELD_NUMBER: builtins.int
    duration_ms: builtins.int
    """In future, the caller will be able to customize when profiling starts and
    stops. For now, it collects `duration_ms` milliseconds worth of data.
    """
    max_events: builtins.int
    """The maximum number of events to return. By default (value 0), return all
    events.
    """
    @property
    def tools(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Required profiling tools name such as "input_pipeline_analyzer" etc"""
    @property
    def tool_options(self) -> google.protobuf.internal.containers.MessageMap[builtins.str, global___ToolRequestOptions]:
        """Specifies the requirement for each tools."""
    @property
    def opts(self) -> tensorflow.core.profiler.profiler_options_pb2.ProfileOptions:
        """Optional profiling options that control how a TF session will be profiled."""
    repository_root: builtins.str
    """The place where we will dump profile data. We will normally use
    MODEL_DIR/plugins/profile/ as the repository root.
    """
    session_id: builtins.str
    """The user provided profile session identifier."""
    host_name: builtins.str
    """The hostname of system where the profile should happen.
    We use it as identifier in part of our output filename.
    """
    def __init__(
        self,
        *,
        duration_ms: builtins.int = ...,
        max_events: builtins.int = ...,
        tools: collections.abc.Iterable[builtins.str] | None = ...,
        tool_options: collections.abc.Mapping[builtins.str, global___ToolRequestOptions] | None = ...,
        opts: tensorflow.core.profiler.profiler_options_pb2.ProfileOptions | None = ...,
        repository_root: builtins.str = ...,
        session_id: builtins.str = ...,
        host_name: builtins.str = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["opts", b"opts"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["duration_ms", b"duration_ms", "host_name", b"host_name", "max_events", b"max_events", "opts", b"opts", "repository_root", b"repository_root", "session_id", b"session_id", "tool_options", b"tool_options", "tools", b"tools"]) -> None: ...

global___ProfileRequest = ProfileRequest

@typing_extensions.final
class ProfileToolData(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    name: builtins.str
    """The file name which this data is associated (e.g. "input_pipeline.json",
    "cluster_xxx.memory_viewer.json").
    """
    data: builtins.bytes
    """The data payload (likely json) for the specific tool."""
    def __init__(
        self,
        *,
        name: builtins.str = ...,
        data: builtins.bytes = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["data", b"data", "name", b"name"]) -> None: ...

global___ProfileToolData = ProfileToolData

@typing_extensions.final
class ProfileResponse(google.protobuf.message.Message):
    """Next-ID: 8"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TOOL_DATA_FIELD_NUMBER: builtins.int
    EMPTY_TRACE_FIELD_NUMBER: builtins.int
    @property
    def tool_data(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ProfileToolData]:
        """Data payload for each required tools."""
    empty_trace: builtins.bool
    """When we write profiling data directly to repository directory, we need a
    way to figure out whether the captured trace is empty.
    """
    def __init__(
        self,
        *,
        tool_data: collections.abc.Iterable[global___ProfileToolData] | None = ...,
        empty_trace: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["empty_trace", b"empty_trace", "tool_data", b"tool_data"]) -> None: ...

global___ProfileResponse = ProfileResponse

@typing_extensions.final
class TerminateRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SESSION_ID_FIELD_NUMBER: builtins.int
    session_id: builtins.str
    """Which session id to terminate."""
    def __init__(
        self,
        *,
        session_id: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["session_id", b"session_id"]) -> None: ...

global___TerminateRequest = TerminateRequest

@typing_extensions.final
class TerminateResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___TerminateResponse = TerminateResponse

@typing_extensions.final
class MonitorRequest(google.protobuf.message.Message):
    """Next-ID: 4"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DURATION_MS_FIELD_NUMBER: builtins.int
    MONITORING_LEVEL_FIELD_NUMBER: builtins.int
    TIMESTAMP_FIELD_NUMBER: builtins.int
    duration_ms: builtins.int
    """Duration for which to profile between each update."""
    monitoring_level: builtins.int
    """Indicates the level at which we want to monitor. Currently, two levels are
    supported:
    Level 1: An ultra lightweight mode that captures only some utilization
    metrics.
    Level 2: More verbose than level 1. Collects utilization metrics, device
    information, step time information, etc. Do not use this option if the TPU
    host is being very heavily used.
    """
    timestamp: builtins.bool
    """True to display timestamp in monitoring result."""
    def __init__(
        self,
        *,
        duration_ms: builtins.int = ...,
        monitoring_level: builtins.int = ...,
        timestamp: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["duration_ms", b"duration_ms", "monitoring_level", b"monitoring_level", "timestamp", b"timestamp"]) -> None: ...

global___MonitorRequest = MonitorRequest

@typing_extensions.final
class MonitorResponse(google.protobuf.message.Message):
    """Next-ID: 11"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DATA_FIELD_NUMBER: builtins.int
    MONITOR_RESULT_FIELD_NUMBER: builtins.int
    data: builtins.str
    """Properly formatted string data that can be directly returned back to user."""
    @property
    def monitor_result(self) -> tensorflow.core.profiler.profiler_service_monitor_result_pb2.ProfilerServiceMonitorResult:
        """A collection of monitoring results for each field show in data."""
    def __init__(
        self,
        *,
        data: builtins.str = ...,
        monitor_result: tensorflow.core.profiler.profiler_service_monitor_result_pb2.ProfilerServiceMonitorResult | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["monitor_result", b"monitor_result"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["data", b"data", "monitor_result", b"monitor_result"]) -> None: ...

global___MonitorResponse = MonitorResponse
