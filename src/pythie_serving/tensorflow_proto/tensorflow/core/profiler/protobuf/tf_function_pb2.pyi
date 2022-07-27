"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _TfFunctionExecutionMode:
    ValueType = typing.NewType('ValueType', builtins.int)
    V: typing_extensions.TypeAlias = ValueType
class _TfFunctionExecutionModeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_TfFunctionExecutionMode.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    INVALID_MODE: _TfFunctionExecutionMode.ValueType  # 0
    """Yet to be set."""

    EAGER_MODE: _TfFunctionExecutionMode.ValueType  # 1
    """Eager execution."""

    TRACED_MODE: _TfFunctionExecutionMode.ValueType  # 2
    """Graph execution with tracing."""

    NOT_TRACED_MODE: _TfFunctionExecutionMode.ValueType  # 3
    """Graph execution without tracing."""

    CONCRETE_MODE: _TfFunctionExecutionMode.ValueType  # 4
    """Concrete function."""

class TfFunctionExecutionMode(_TfFunctionExecutionMode, metaclass=_TfFunctionExecutionModeEnumTypeWrapper):
    """All possible execution modes of a tf-function."""
    pass

INVALID_MODE: TfFunctionExecutionMode.ValueType  # 0
"""Yet to be set."""

EAGER_MODE: TfFunctionExecutionMode.ValueType  # 1
"""Eager execution."""

TRACED_MODE: TfFunctionExecutionMode.ValueType  # 2
"""Graph execution with tracing."""

NOT_TRACED_MODE: TfFunctionExecutionMode.ValueType  # 3
"""Graph execution without tracing."""

CONCRETE_MODE: TfFunctionExecutionMode.ValueType  # 4
"""Concrete function."""

global___TfFunctionExecutionMode = TfFunctionExecutionMode


class _TfFunctionCompiler:
    ValueType = typing.NewType('ValueType', builtins.int)
    V: typing_extensions.TypeAlias = ValueType
class _TfFunctionCompilerEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_TfFunctionCompiler.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    INVALID_COMPILER: _TfFunctionCompiler.ValueType  # 0
    """Yet to be set."""

    OTHER_COMPILER: _TfFunctionCompiler.ValueType  # 1
    """Any other compiler."""

    MIXED_COMPILER: _TfFunctionCompiler.ValueType  # 2
    """If some instance of the function is compiled with XLA and some is compiled
    with Non-XLA, use "MIXED_COMPILER".
    """

    XLA_COMPILER: _TfFunctionCompiler.ValueType  # 3
    """XLA compiler."""

    MLIR_COMPILER: _TfFunctionCompiler.ValueType  # 4
    """MLIR compiler."""

class TfFunctionCompiler(_TfFunctionCompiler, metaclass=_TfFunctionCompilerEnumTypeWrapper):
    """All possible compilers that can be used to compile a tf-function in the graph
    mode.
    """
    pass

INVALID_COMPILER: TfFunctionCompiler.ValueType  # 0
"""Yet to be set."""

OTHER_COMPILER: TfFunctionCompiler.ValueType  # 1
"""Any other compiler."""

MIXED_COMPILER: TfFunctionCompiler.ValueType  # 2
"""If some instance of the function is compiled with XLA and some is compiled
with Non-XLA, use "MIXED_COMPILER".
"""

XLA_COMPILER: TfFunctionCompiler.ValueType  # 3
"""XLA compiler."""

MLIR_COMPILER: TfFunctionCompiler.ValueType  # 4
"""MLIR compiler."""

global___TfFunctionCompiler = TfFunctionCompiler


class TfFunctionMetrics(google.protobuf.message.Message):
    """Metrics associated with a particular execution mode of a tf-function."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    COUNT_FIELD_NUMBER: builtins.int
    SELF_TIME_PS_FIELD_NUMBER: builtins.int
    count: builtins.int
    """Number of invocations to the function in that execution mode."""

    self_time_ps: builtins.int
    """The sum of "self-execution" time of this function over those invocations."""

    def __init__(self,
        *,
        count: builtins.int = ...,
        self_time_ps: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["count",b"count","self_time_ps",b"self_time_ps"]) -> None: ...
global___TfFunctionMetrics = TfFunctionMetrics

class TfFunction(google.protobuf.message.Message):
    """Statistics for a tf-function."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class MetricsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.int
        @property
        def value(self) -> global___TfFunctionMetrics: ...
        def __init__(self,
            *,
            key: builtins.int = ...,
            value: typing.Optional[global___TfFunctionMetrics] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    METRICS_FIELD_NUMBER: builtins.int
    TOTAL_TRACING_COUNT_FIELD_NUMBER: builtins.int
    COMPILER_FIELD_NUMBER: builtins.int
    EXPENSIVE_CALL_PERCENT_FIELD_NUMBER: builtins.int
    @property
    def metrics(self) -> google.protobuf.internal.containers.MessageMap[builtins.int, global___TfFunctionMetrics]:
        """A map from each execution mode to its corresponding metrics."""
        pass
    total_tracing_count: builtins.int
    """Total tracing count from the program's beginning (i.e. beyond the profiling
    period) of this tf-function.
    """

    compiler: global___TfFunctionCompiler.ValueType
    """Compiler used to compile this function."""

    expensive_call_percent: builtins.float
    """Percentage of time spent in the expensive calls to this function in the
    profiled period.
    """

    def __init__(self,
        *,
        metrics: typing.Optional[typing.Mapping[builtins.int, global___TfFunctionMetrics]] = ...,
        total_tracing_count: builtins.int = ...,
        compiler: global___TfFunctionCompiler.ValueType = ...,
        expensive_call_percent: builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["compiler",b"compiler","expensive_call_percent",b"expensive_call_percent","metrics",b"metrics","total_tracing_count",b"total_tracing_count"]) -> None: ...
global___TfFunction = TfFunction

class TfFunctionDb(google.protobuf.message.Message):
    """Statistics for all tf-functions."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class TfFunctionsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> global___TfFunction: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[global___TfFunction] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    TF_FUNCTIONS_FIELD_NUMBER: builtins.int
    @property
    def tf_functions(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___TfFunction]:
        """A map from function name to the statistics of that function."""
        pass
    def __init__(self,
        *,
        tf_functions: typing.Optional[typing.Mapping[typing.Text, global___TfFunction]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["tf_functions",b"tf_functions"]) -> None: ...
global___TfFunctionDb = TfFunctionDb