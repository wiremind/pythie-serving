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
class BackendConfig(google.protobuf.message.Message):
    """Backend config for XLA:CPU."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    OUTER_DIMENSION_PARTITIONS_FIELD_NUMBER: builtins.int
    @property
    def outer_dimension_partitions(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Number of partitions per outer dimension (in order, starting with
        outer-most dimension first). Used by the parallel cpu backend to partition
        HLOs into parallel tasks.
        """
    def __init__(
        self,
        *,
        outer_dimension_partitions: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["outer_dimension_partitions", b"outer_dimension_partitions"]) -> None: ...

global___BackendConfig = BackendConfig
