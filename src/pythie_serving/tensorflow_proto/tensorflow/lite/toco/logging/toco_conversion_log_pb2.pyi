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
class TocoConversionLog(google.protobuf.message.Message):
    """TocoConversionLog contains the analytics to be gathered when user converts
    a model to TF Lite using TOCO.
    Next ID to USE: 14.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class BuiltInOpsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.int
        def __init__(
            self,
            *,
            key: builtins.str | None = ...,
            value: builtins.int | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing_extensions.final
    class CustomOpsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.int
        def __init__(
            self,
            *,
            key: builtins.str | None = ...,
            value: builtins.int | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing_extensions.final
    class SelectOpsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.int
        def __init__(
            self,
            *,
            key: builtins.str | None = ...,
            value: builtins.int | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    OP_LIST_FIELD_NUMBER: builtins.int
    BUILT_IN_OPS_FIELD_NUMBER: builtins.int
    CUSTOM_OPS_FIELD_NUMBER: builtins.int
    SELECT_OPS_FIELD_NUMBER: builtins.int
    OP_SIGNATURES_FIELD_NUMBER: builtins.int
    INPUT_TENSOR_TYPES_FIELD_NUMBER: builtins.int
    OUTPUT_TENSOR_TYPES_FIELD_NUMBER: builtins.int
    LOG_GENERATION_TS_FIELD_NUMBER: builtins.int
    MODEL_SIZE_FIELD_NUMBER: builtins.int
    TF_LITE_VERSION_FIELD_NUMBER: builtins.int
    OS_VERSION_FIELD_NUMBER: builtins.int
    MODEL_HASH_FIELD_NUMBER: builtins.int
    TOCO_ERR_LOGS_FIELD_NUMBER: builtins.int
    @property
    def op_list(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Total ops listed by name."""
    @property
    def built_in_ops(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.int]:
        """Counts of built-in ops.
        Key is op name and value is the count.
        """
    @property
    def custom_ops(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.int]:
        """Counts of custom ops."""
    @property
    def select_ops(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.int]:
        """Counts of select ops."""
    @property
    def op_signatures(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """The signature of operators. Including ops input/output types and shapes,
        op name and version.
        """
    @property
    def input_tensor_types(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Input tensor types."""
    @property
    def output_tensor_types(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]:
        """Output tensor types."""
    log_generation_ts: builtins.int
    """Log generation time in micro-seconds."""
    model_size: builtins.int
    """Total number of ops in the model."""
    tf_lite_version: builtins.str
    """Tensorflow Lite runtime version."""
    os_version: builtins.str
    """Operating System info."""
    model_hash: builtins.str
    """Model hash string."""
    toco_err_logs: builtins.str
    """Error messages emitted by TOCO during conversion."""
    def __init__(
        self,
        *,
        op_list: collections.abc.Iterable[builtins.str] | None = ...,
        built_in_ops: collections.abc.Mapping[builtins.str, builtins.int] | None = ...,
        custom_ops: collections.abc.Mapping[builtins.str, builtins.int] | None = ...,
        select_ops: collections.abc.Mapping[builtins.str, builtins.int] | None = ...,
        op_signatures: collections.abc.Iterable[builtins.str] | None = ...,
        input_tensor_types: collections.abc.Iterable[builtins.str] | None = ...,
        output_tensor_types: collections.abc.Iterable[builtins.str] | None = ...,
        log_generation_ts: builtins.int | None = ...,
        model_size: builtins.int | None = ...,
        tf_lite_version: builtins.str | None = ...,
        os_version: builtins.str | None = ...,
        model_hash: builtins.str | None = ...,
        toco_err_logs: builtins.str | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["log_generation_ts", b"log_generation_ts", "model_hash", b"model_hash", "model_size", b"model_size", "os_version", b"os_version", "tf_lite_version", b"tf_lite_version", "toco_err_logs", b"toco_err_logs"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["built_in_ops", b"built_in_ops", "custom_ops", b"custom_ops", "input_tensor_types", b"input_tensor_types", "log_generation_ts", b"log_generation_ts", "model_hash", b"model_hash", "model_size", b"model_size", "op_list", b"op_list", "op_signatures", b"op_signatures", "os_version", b"os_version", "output_tensor_types", b"output_tensor_types", "select_ops", b"select_ops", "tf_lite_version", b"tf_lite_version", "toco_err_logs", b"toco_err_logs"]) -> None: ...

global___TocoConversionLog = TocoConversionLog
