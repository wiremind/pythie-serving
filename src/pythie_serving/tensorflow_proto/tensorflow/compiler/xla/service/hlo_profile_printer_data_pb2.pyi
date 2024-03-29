"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class HloProfilePrinterData(google.protobuf.message.Message):
    """Describes how to pretty-print a profile counter array gathered for a specific
    HloModule.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class HloInstructionInfo(google.protobuf.message.Message):
        """Pretty-printer information about an HloInstruction."""

        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        LONG_NAME_FIELD_NUMBER: builtins.int
        SHORT_NAME_FIELD_NUMBER: builtins.int
        CATEGORY_FIELD_NUMBER: builtins.int
        FLOP_COUNT_FIELD_NUMBER: builtins.int
        TRANSCENDENTAL_COUNT_FIELD_NUMBER: builtins.int
        BYTES_ACCESSED_FIELD_NUMBER: builtins.int
        OPTIMAL_SECONDS_FIELD_NUMBER: builtins.int
        PROFILE_INDEX_FIELD_NUMBER: builtins.int
        long_name: builtins.str
        short_name: builtins.str
        category: builtins.str
        flop_count: builtins.float
        """Metrics computed by HloCostAnalysis."""
        transcendental_count: builtins.float
        bytes_accessed: builtins.int
        optimal_seconds: builtins.float
        profile_index: builtins.int
        """The index into the profile counters array for the HloInstruction
        corresponding to this HloInstructionInfo.
        """
        def __init__(
            self,
            *,
            long_name: builtins.str = ...,
            short_name: builtins.str = ...,
            category: builtins.str = ...,
            flop_count: builtins.float = ...,
            transcendental_count: builtins.float = ...,
            bytes_accessed: builtins.int = ...,
            optimal_seconds: builtins.float = ...,
            profile_index: builtins.int = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["bytes_accessed", b"bytes_accessed", "category", b"category", "flop_count", b"flop_count", "long_name", b"long_name", "optimal_seconds", b"optimal_seconds", "profile_index", b"profile_index", "short_name", b"short_name", "transcendental_count", b"transcendental_count"]) -> None: ...

    @typing_extensions.final
    class HloComputationInfo(google.protobuf.message.Message):
        """Pretty-printer information about an HloComputation."""

        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        NAME_FIELD_NUMBER: builtins.int
        PROFILE_INDEX_FIELD_NUMBER: builtins.int
        INSTRUCTION_INFOS_FIELD_NUMBER: builtins.int
        name: builtins.str
        profile_index: builtins.int
        """The index into the profile counters array for the HloComputation
        corresponding to this HloComputationInfo.
        """
        @property
        def instruction_infos(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___HloProfilePrinterData.HloInstructionInfo]:
            """HloInstructionInfos for every HloInstruction in the HloComputation for
            corresponding to this HloComputattionInfo.
            """
        def __init__(
            self,
            *,
            name: builtins.str = ...,
            profile_index: builtins.int = ...,
            instruction_infos: collections.abc.Iterable[global___HloProfilePrinterData.HloInstructionInfo] | None = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["instruction_infos", b"instruction_infos", "name", b"name", "profile_index", b"profile_index"]) -> None: ...

    @typing_extensions.final
    class ExtraMetricsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.int
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.int = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    COMPUTATION_INFOS_FIELD_NUMBER: builtins.int
    PROFILE_COUNTERS_SIZE_FIELD_NUMBER: builtins.int
    EXTRA_METRICS_FIELD_NUMBER: builtins.int
    ENTRY_COMPUTATION_FIELD_NUMBER: builtins.int
    @property
    def computation_infos(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___HloProfilePrinterData.HloComputationInfo]:
        """HloComputationInfos for every HloComputation in the HloModule."""
    profile_counters_size: builtins.int
    """The size of the profile counters array we will pretty-print."""
    @property
    def extra_metrics(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.int]:
        """Maps extra metric name to the index into the profile counters array."""
    entry_computation: builtins.str
    """Name of the entry computation."""
    def __init__(
        self,
        *,
        computation_infos: collections.abc.Iterable[global___HloProfilePrinterData.HloComputationInfo] | None = ...,
        profile_counters_size: builtins.int = ...,
        extra_metrics: collections.abc.Mapping[builtins.str, builtins.int] | None = ...,
        entry_computation: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["computation_infos", b"computation_infos", "entry_computation", b"entry_computation", "extra_metrics", b"extra_metrics", "profile_counters_size", b"profile_counters_size"]) -> None: ...

global___HloProfilePrinterData = HloProfilePrinterData
