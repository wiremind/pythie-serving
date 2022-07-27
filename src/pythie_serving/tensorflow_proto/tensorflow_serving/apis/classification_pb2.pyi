"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow_serving.apis.input_pb2
import tensorflow_serving.apis.model_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Class(google.protobuf.message.Message):
    """A single class."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    LABEL_FIELD_NUMBER: builtins.int
    SCORE_FIELD_NUMBER: builtins.int
    label: typing.Text
    """Label or name of the class."""

    score: builtins.float
    """Score for this class (e.g., the probability the item belongs to this
    class). As per the proto3 default-value semantics, if the score is missing,
    it should be treated as 0.
    """

    def __init__(self,
        *,
        label: typing.Text = ...,
        score: builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["label",b"label","score",b"score"]) -> None: ...
global___Class = Class

class Classifications(google.protobuf.message.Message):
    """List of classes for a single item (tensorflow.Example)."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    CLASSES_FIELD_NUMBER: builtins.int
    @property
    def classes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Class]: ...
    def __init__(self,
        *,
        classes: typing.Optional[typing.Iterable[global___Class]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["classes",b"classes"]) -> None: ...
global___Classifications = Classifications

class ClassificationResult(google.protobuf.message.Message):
    """Contains one result per input example, in the same order as the input in
    ClassificationRequest.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    CLASSIFICATIONS_FIELD_NUMBER: builtins.int
    @property
    def classifications(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Classifications]: ...
    def __init__(self,
        *,
        classifications: typing.Optional[typing.Iterable[global___Classifications]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["classifications",b"classifications"]) -> None: ...
global___ClassificationResult = ClassificationResult

class ClassificationRequest(google.protobuf.message.Message):
    """RPC Interfaces

    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    MODEL_SPEC_FIELD_NUMBER: builtins.int
    INPUT_FIELD_NUMBER: builtins.int
    @property
    def model_spec(self) -> tensorflow_serving.apis.model_pb2.ModelSpec:
        """Model Specification. If version is not specified, will use the latest
        (numerical) version.
        """
        pass
    @property
    def input(self) -> tensorflow_serving.apis.input_pb2.Input:
        """Input data."""
        pass
    def __init__(self,
        *,
        model_spec: typing.Optional[tensorflow_serving.apis.model_pb2.ModelSpec] = ...,
        input: typing.Optional[tensorflow_serving.apis.input_pb2.Input] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["input",b"input","model_spec",b"model_spec"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["input",b"input","model_spec",b"model_spec"]) -> None: ...
global___ClassificationRequest = ClassificationRequest

class ClassificationResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    MODEL_SPEC_FIELD_NUMBER: builtins.int
    RESULT_FIELD_NUMBER: builtins.int
    @property
    def model_spec(self) -> tensorflow_serving.apis.model_pb2.ModelSpec:
        """Effective Model Specification used for classification."""
        pass
    @property
    def result(self) -> global___ClassificationResult:
        """Result of the classification."""
        pass
    def __init__(self,
        *,
        model_spec: typing.Optional[tensorflow_serving.apis.model_pb2.ModelSpec] = ...,
        result: typing.Optional[global___ClassificationResult] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["model_spec",b"model_spec","result",b"result"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["model_spec",b"model_spec","result",b"result"]) -> None: ...
global___ClassificationResponse = ClassificationResponse
