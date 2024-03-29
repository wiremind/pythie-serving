"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
import google.protobuf.descriptor
import google.protobuf.message
import sys
import tensorflow.lite.tools.evaluation.proto.evaluation_stages_pb2

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class EvaluationStageConfig(google.protobuf.message.Message):
    """Contains parameters that define how an EvaluationStage will be executed.
    This would typically be validated only once during initialization, so should
    not contain any variables that change with each run.

    Next ID: 3
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    SPECIFICATION_FIELD_NUMBER: builtins.int
    name: builtins.str
    @property
    def specification(self) -> tensorflow.lite.tools.evaluation.proto.evaluation_stages_pb2.ProcessSpecification:
        """Specification defining what this stage does, and any required parameters."""
    def __init__(
        self,
        *,
        name: builtins.str | None = ...,
        specification: tensorflow.lite.tools.evaluation.proto.evaluation_stages_pb2.ProcessSpecification | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["name", b"name", "specification", b"specification"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["name", b"name", "specification", b"specification"]) -> None: ...

global___EvaluationStageConfig = EvaluationStageConfig

@typing_extensions.final
class EvaluationStageMetrics(google.protobuf.message.Message):
    """Metrics returned from EvaluationStage.LatestMetrics() need not have all
    fields set.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUM_RUNS_FIELD_NUMBER: builtins.int
    PROCESS_METRICS_FIELD_NUMBER: builtins.int
    num_runs: builtins.int
    """Total number of times the EvaluationStage is run."""
    @property
    def process_metrics(self) -> tensorflow.lite.tools.evaluation.proto.evaluation_stages_pb2.ProcessMetrics:
        """Process-specific numbers such as latencies, accuracy, etc."""
    def __init__(
        self,
        *,
        num_runs: builtins.int | None = ...,
        process_metrics: tensorflow.lite.tools.evaluation.proto.evaluation_stages_pb2.ProcessMetrics | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["num_runs", b"num_runs", "process_metrics", b"process_metrics"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["num_runs", b"num_runs", "process_metrics", b"process_metrics"]) -> None: ...

global___EvaluationStageMetrics = EvaluationStageMetrics
