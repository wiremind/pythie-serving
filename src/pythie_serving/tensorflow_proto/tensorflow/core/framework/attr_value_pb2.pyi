"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow.core.framework.tensor_pb2
import tensorflow.core.framework.tensor_shape_pb2
import tensorflow.core.framework.types_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class AttrValue(google.protobuf.message.Message):
    """Protocol buffer representing the value for an attr used to configure an Op.
    Comment indicates the corresponding attr type.  Only the field matching the
    attr type may be filled.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class ListValue(google.protobuf.message.Message):
        """LINT.IfChange"""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        S_FIELD_NUMBER: builtins.int
        I_FIELD_NUMBER: builtins.int
        F_FIELD_NUMBER: builtins.int
        B_FIELD_NUMBER: builtins.int
        TYPE_FIELD_NUMBER: builtins.int
        SHAPE_FIELD_NUMBER: builtins.int
        TENSOR_FIELD_NUMBER: builtins.int
        FUNC_FIELD_NUMBER: builtins.int
        @property
        def s(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bytes]:
            """"list(string)" """
            pass
        @property
        def i(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
            """"list(int)" """
            pass
        @property
        def f(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
            """"list(float)" """
            pass
        @property
        def b(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bool]:
            """"list(bool)" """
            pass
        @property
        def type(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[tensorflow.core.framework.types_pb2.DataType.ValueType]:
            """"list(type)" """
            pass
        @property
        def shape(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto]:
            """"list(shape)" """
            pass
        @property
        def tensor(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[tensorflow.core.framework.tensor_pb2.TensorProto]:
            """"list(tensor)" """
            pass
        @property
        def func(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___NameAttrList]:
            """"list(attr)" """
            pass
        def __init__(self,
            *,
            s: typing.Optional[typing.Iterable[builtins.bytes]] = ...,
            i: typing.Optional[typing.Iterable[builtins.int]] = ...,
            f: typing.Optional[typing.Iterable[builtins.float]] = ...,
            b: typing.Optional[typing.Iterable[builtins.bool]] = ...,
            type: typing.Optional[typing.Iterable[tensorflow.core.framework.types_pb2.DataType.ValueType]] = ...,
            shape: typing.Optional[typing.Iterable[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto]] = ...,
            tensor: typing.Optional[typing.Iterable[tensorflow.core.framework.tensor_pb2.TensorProto]] = ...,
            func: typing.Optional[typing.Iterable[global___NameAttrList]] = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["b",b"b","f",b"f","func",b"func","i",b"i","s",b"s","shape",b"shape","tensor",b"tensor","type",b"type"]) -> None: ...

    S_FIELD_NUMBER: builtins.int
    I_FIELD_NUMBER: builtins.int
    F_FIELD_NUMBER: builtins.int
    B_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    SHAPE_FIELD_NUMBER: builtins.int
    TENSOR_FIELD_NUMBER: builtins.int
    LIST_FIELD_NUMBER: builtins.int
    FUNC_FIELD_NUMBER: builtins.int
    PLACEHOLDER_FIELD_NUMBER: builtins.int
    s: builtins.bytes
    """"string" """

    i: builtins.int
    """"int" """

    f: builtins.float
    """"float" """

    b: builtins.bool
    """"bool" """

    type: tensorflow.core.framework.types_pb2.DataType.ValueType
    """"type" """

    @property
    def shape(self) -> tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto:
        """"shape" """
        pass
    @property
    def tensor(self) -> tensorflow.core.framework.tensor_pb2.TensorProto:
        """"tensor" """
        pass
    @property
    def list(self) -> global___AttrValue.ListValue:
        """any "list(...)" """
        pass
    @property
    def func(self) -> global___NameAttrList:
        """"func" represents a function. func.name is a function's name or
        a primitive op's name. func.attr.first is the name of an attr
        defined for that function. func.attr.second is the value for
        that attr in the instantiation.
        """
        pass
    placeholder: typing.Text
    """This is a placeholder only used in nodes defined inside a
    function.  It indicates the attr value will be supplied when
    the function is instantiated.  For example, let us suppose a
    node "N" in function "FN". "N" has an attr "A" with value
    placeholder = "foo". When FN is instantiated with attr "foo"
    set to "bar", the instantiated node N's attr A will have been
    given the value "bar".
    """

    def __init__(self,
        *,
        s: builtins.bytes = ...,
        i: builtins.int = ...,
        f: builtins.float = ...,
        b: builtins.bool = ...,
        type: tensorflow.core.framework.types_pb2.DataType.ValueType = ...,
        shape: typing.Optional[tensorflow.core.framework.tensor_shape_pb2.TensorShapeProto] = ...,
        tensor: typing.Optional[tensorflow.core.framework.tensor_pb2.TensorProto] = ...,
        list: typing.Optional[global___AttrValue.ListValue] = ...,
        func: typing.Optional[global___NameAttrList] = ...,
        placeholder: typing.Text = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["b",b"b","f",b"f","func",b"func","i",b"i","list",b"list","placeholder",b"placeholder","s",b"s","shape",b"shape","tensor",b"tensor","type",b"type","value",b"value"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["b",b"b","f",b"f","func",b"func","i",b"i","list",b"list","placeholder",b"placeholder","s",b"s","shape",b"shape","tensor",b"tensor","type",b"type","value",b"value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["value",b"value"]) -> typing.Optional[typing_extensions.Literal["s","i","f","b","type","shape","tensor","list","func","placeholder"]]: ...
global___AttrValue = AttrValue

class NameAttrList(google.protobuf.message.Message):
    """A list of attr names and their values. The whole list is attached
    with a string name.  E.g., MatMul[T=float].
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class AttrEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: typing.Text
        @property
        def value(self) -> global___AttrValue: ...
        def __init__(self,
            *,
            key: typing.Text = ...,
            value: typing.Optional[global___AttrValue] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["value",b"value"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["key",b"key","value",b"value"]) -> None: ...

    NAME_FIELD_NUMBER: builtins.int
    ATTR_FIELD_NUMBER: builtins.int
    name: typing.Text
    @property
    def attr(self) -> google.protobuf.internal.containers.MessageMap[typing.Text, global___AttrValue]: ...
    def __init__(self,
        *,
        name: typing.Text = ...,
        attr: typing.Optional[typing.Mapping[typing.Text, global___AttrValue]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["attr",b"attr","name",b"name"]) -> None: ...
global___NameAttrList = NameAttrList