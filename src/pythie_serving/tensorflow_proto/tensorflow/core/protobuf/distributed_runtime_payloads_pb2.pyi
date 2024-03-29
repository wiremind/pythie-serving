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

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class GrpcPayloadContainer(google.protobuf.message.Message):
    """Used to serialize and transmit tensorflow::Status payloads through
    grpc::Status `error_details` since grpc::Status lacks payload API.
    TODO(b/204231601): Use GRPC API once supported.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing_extensions.final
    class PayloadsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.bytes
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.bytes = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    PAYLOADS_FIELD_NUMBER: builtins.int
    @property
    def payloads(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.bytes]: ...
    def __init__(
        self,
        *,
        payloads: collections.abc.Mapping[builtins.str, builtins.bytes] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["payloads", b"payloads"]) -> None: ...

global___GrpcPayloadContainer = GrpcPayloadContainer

@typing_extensions.final
class GrpcPayloadsLost(google.protobuf.message.Message):
    """If included as a payload, this message flags the Status to have lost payloads
    during the GRPC transmission.
    URI: "type.googleapis.com/tensorflow.distributed_runtime.GrpcPayloadsLost"
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___GrpcPayloadsLost = GrpcPayloadsLost

@typing_extensions.final
class WorkerPossiblyRestarted(google.protobuf.message.Message):
    """If included as a payload, this message flags the Status to be a possible
    outcome of a worker restart.
    URI:
    "type.googleapis.com/tensorflow.distributed_runtime.WorkerPossiblyRestarted"
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___WorkerPossiblyRestarted = WorkerPossiblyRestarted
