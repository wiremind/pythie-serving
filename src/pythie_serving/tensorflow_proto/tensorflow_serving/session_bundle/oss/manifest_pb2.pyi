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

class Signatures(google.protobuf.message.Message):
    """Signatures of model export."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class NamedSignaturesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> global___Signature: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[global___Signature] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    DEFAULT_SIGNATURE_FIELD_NUMBER: builtins.int
    NAMED_SIGNATURES_FIELD_NUMBER: builtins.int
    @property
    def default_signature(self) -> global___Signature:
        """Default signature of the graph.
        WARNING(break-tutorial-inline-code): The following code snippet is
        in-lined in tutorials, please update tutorial documents accordingly
        whenever code changes.
        """
        pass
    @property
    def named_signatures(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___Signature]:
        """Named signatures of the graph."""
        pass
    def __init__(self,
        *,
        default_signature: typing.Optional[global___Signature] = ...,
        named_signatures: typing.Optional[typing.Mapping[typing.Text, global___Signature]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["default_signature",b"default_signature"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["default_signature",b"default_signature","named_signatures",b"named_signatures"]) -> None: ...
global___Signatures = Signatures

class TensorBinding(google.protobuf.message.Message):
    """A binding to a tensor including the name and, possibly in the future, type
    or other metadata. For example, this may specify whether a tensor supports
    batch vs single inference.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TENSOR_NAME_FIELD_NUMBER: builtins.int
    tensor_name: typing.Text
    """The name of the tensor to bind to."""

    def __init__(self,
        *,
        tensor_name: typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["tensor_name",b"tensor_name"]) -> None: ...
global___TensorBinding = TensorBinding

class AssetFile(google.protobuf.message.Message):
    """An asset file or set of sharded files with the same name that will be bound
    to a tensor at init / session_bundle load time.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TENSOR_BINDING_FIELD_NUMBER: builtins.int
    FILENAME_FIELD_NUMBER: builtins.int
    @property
    def tensor_binding(self) -> global___TensorBinding:
        """The tensor to bind the asset filename to."""
        pass
    filename: typing.Text
    """The filename within the assets directory. Note: does not include the base
    path or asset directory prefix. Base paths can and will change when models
    are deployed for serving.
    """

    def __init__(self,
        *,
        tensor_binding: typing.Optional[global___TensorBinding] = ...,
        filename: typing.Text = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["tensor_binding",b"tensor_binding"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["filename",b"filename","tensor_binding",b"tensor_binding"]) -> None: ...
global___AssetFile = AssetFile

class Signature(google.protobuf.message.Message):
    """A Signature specifies the inputs and outputs of commonly used graphs."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    REGRESSION_SIGNATURE_FIELD_NUMBER: builtins.int
    CLASSIFICATION_SIGNATURE_FIELD_NUMBER: builtins.int
    GENERIC_SIGNATURE_FIELD_NUMBER: builtins.int
    @property
    def regression_signature(self) -> global___RegressionSignature: ...
    @property
    def classification_signature(self) -> global___ClassificationSignature: ...
    @property
    def generic_signature(self) -> global___GenericSignature: ...
    def __init__(self,
        *,
        regression_signature: typing.Optional[global___RegressionSignature] = ...,
        classification_signature: typing.Optional[global___ClassificationSignature] = ...,
        generic_signature: typing.Optional[global___GenericSignature] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["classification_signature",b"classification_signature","generic_signature",b"generic_signature","regression_signature",b"regression_signature","type",b"type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["classification_signature",b"classification_signature","generic_signature",b"generic_signature","regression_signature",b"regression_signature","type",b"type"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["type",b"type"]) -> typing.Optional[typing_extensions.Literal["regression_signature","classification_signature","generic_signature"]]: ...
global___Signature = Signature

class RegressionSignature(google.protobuf.message.Message):
    """RegressionSignature specifies a graph that takes an input and returns an
    output.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    INPUT_FIELD_NUMBER: builtins.int
    OUTPUT_FIELD_NUMBER: builtins.int
    @property
    def input(self) -> global___TensorBinding: ...
    @property
    def output(self) -> global___TensorBinding: ...
    def __init__(self,
        *,
        input: typing.Optional[global___TensorBinding] = ...,
        output: typing.Optional[global___TensorBinding] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["input",b"input","output",b"output"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["input",b"input","output",b"output"]) -> None: ...
global___RegressionSignature = RegressionSignature

class ClassificationSignature(google.protobuf.message.Message):
    """ClassificationSignature specifies a graph that takes an input and returns
    classes and their scores.
    WARNING(break-tutorial-inline-code): The following code snippet is
    in-lined in tutorials, please update tutorial documents accordingly
    whenever code changes.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    INPUT_FIELD_NUMBER: builtins.int
    CLASSES_FIELD_NUMBER: builtins.int
    SCORES_FIELD_NUMBER: builtins.int
    @property
    def input(self) -> global___TensorBinding: ...
    @property
    def classes(self) -> global___TensorBinding: ...
    @property
    def scores(self) -> global___TensorBinding: ...
    def __init__(self,
        *,
        input: typing.Optional[global___TensorBinding] = ...,
        classes: typing.Optional[global___TensorBinding] = ...,
        scores: typing.Optional[global___TensorBinding] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["classes",b"classes","input",b"input","scores",b"scores"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["classes",b"classes","input",b"input","scores",b"scores"]) -> None: ...
global___ClassificationSignature = ClassificationSignature

class GenericSignature(google.protobuf.message.Message):
    """GenericSignature specifies a map from logical name to Tensor name.
    Typical application of GenericSignature is to use a single GenericSignature
    that includes all of the Tensor nodes and target names that may be useful at
    serving, analysis or debugging time. The recommended name for this signature
    in the ModelManifest is "generic_bindings".
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class MapEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> global___TensorBinding: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[global___TensorBinding] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    MAP_FIELD_NUMBER: builtins.int
    @property
    def map(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___TensorBinding]: ...
    def __init__(self,
        *,
        map: typing.Optional[typing.Mapping[typing.Text, global___TensorBinding]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["map",b"map"]) -> None: ...
global___GenericSignature = GenericSignature