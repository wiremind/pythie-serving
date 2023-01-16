"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
This file contains messages for various machine learning inferences
such as regression and classification.

In many applications more than one type of inference is desired for a single
input.  For example, given meteorologic data an application may want to
perform a classification to determine if we should expect rain, snow or sun
and also perform a regression to predict the temperature.
Sharing the single input data between two inference tasks can be accomplished
using MultiInferenceRequest and MultiInferenceResponse.
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys
import tensorflow_serving.apis.classification_pb2
import tensorflow_serving.apis.input_pb2
import tensorflow_serving.apis.model_pb2
import tensorflow_serving.apis.regression_pb2

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class InferenceTask(google.protobuf.message.Message):
    """Inference request such as classification, regression, etc..."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_SPEC_FIELD_NUMBER: builtins.int
    METHOD_NAME_FIELD_NUMBER: builtins.int
    @property
    def model_spec(self) -> tensorflow_serving.apis.model_pb2.ModelSpec:
        """Model Specification. If version is not specified, will use the latest
        (numerical) version.
        All ModelSpecs in a MultiInferenceRequest must access the same model name.
        """
    method_name: builtins.str
    """Signature's method_name. Should be one of the method names defined in
    third_party/tensorflow/python/saved_model/signature_constants.py.
    e.g. "tensorflow/serving/classify".
    """
    def __init__(
        self,
        *,
        model_spec: tensorflow_serving.apis.model_pb2.ModelSpec | None = ...,
        method_name: builtins.str = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["model_spec", b"model_spec"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["method_name", b"method_name", "model_spec", b"model_spec"]) -> None: ...

global___InferenceTask = InferenceTask

@typing_extensions.final
class InferenceResult(google.protobuf.message.Message):
    """Inference result, matches the type of request or is an error."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MODEL_SPEC_FIELD_NUMBER: builtins.int
    CLASSIFICATION_RESULT_FIELD_NUMBER: builtins.int
    REGRESSION_RESULT_FIELD_NUMBER: builtins.int
    @property
    def model_spec(self) -> tensorflow_serving.apis.model_pb2.ModelSpec: ...
    @property
    def classification_result(self) -> tensorflow_serving.apis.classification_pb2.ClassificationResult: ...
    @property
    def regression_result(self) -> tensorflow_serving.apis.regression_pb2.RegressionResult: ...
    def __init__(
        self,
        *,
        model_spec: tensorflow_serving.apis.model_pb2.ModelSpec | None = ...,
        classification_result: tensorflow_serving.apis.classification_pb2.ClassificationResult | None = ...,
        regression_result: tensorflow_serving.apis.regression_pb2.RegressionResult | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["classification_result", b"classification_result", "model_spec", b"model_spec", "regression_result", b"regression_result", "result", b"result"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["classification_result", b"classification_result", "model_spec", b"model_spec", "regression_result", b"regression_result", "result", b"result"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["result", b"result"]) -> typing_extensions.Literal["classification_result", "regression_result"] | None: ...

global___InferenceResult = InferenceResult

@typing_extensions.final
class MultiInferenceRequest(google.protobuf.message.Message):
    """Inference request containing one or more requests."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TASKS_FIELD_NUMBER: builtins.int
    INPUT_FIELD_NUMBER: builtins.int
    @property
    def tasks(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___InferenceTask]:
        """Inference tasks."""
    @property
    def input(self) -> tensorflow_serving.apis.input_pb2.Input:
        """Input data."""
    def __init__(
        self,
        *,
        tasks: collections.abc.Iterable[global___InferenceTask] | None = ...,
        input: tensorflow_serving.apis.input_pb2.Input | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["input", b"input"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["input", b"input", "tasks", b"tasks"]) -> None: ...

global___MultiInferenceRequest = MultiInferenceRequest

@typing_extensions.final
class MultiInferenceResponse(google.protobuf.message.Message):
    """Inference request containing one or more responses."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    RESULTS_FIELD_NUMBER: builtins.int
    @property
    def results(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___InferenceResult]:
        """List of results; one for each InferenceTask in the request, returned in the
        same order as the request.
        """
    def __init__(
        self,
        *,
        results: collections.abc.Iterable[global___InferenceResult] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["results", b"results"]) -> None: ...

global___MultiInferenceResponse = MultiInferenceResponse
