# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/compiler/tf2xla/tf2xla.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.framework import types_pb2 as tensorflow_dot_core_dot_framework_dot_types__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\'tensorflow/compiler/tf2xla/tf2xla.proto\x12\x11tensorflow.tf2xla\x1a,tensorflow/core/framework/tensor_shape.proto\x1a%tensorflow/core/framework/types.proto\"3\n\x08TensorId\x12\x11\n\tnode_name\x18\x01 \x01(\t\x12\x14\n\x0coutput_index\x18\x02 \x01(\x03\"\x8e\x01\n\x04\x46\x65\x65\x64\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x1b.tensorflow.tf2xla.TensorId\x12+\n\x05shape\x18\x02 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\"\n\x04type\x18\x04 \x01(\x0e\x32\x14.tensorflow.DataType\"\x8f\x01\n\x05\x46\x65tch\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x1b.tensorflow.tf2xla.TensorId\x12\x0c\n\x04name\x18\x02 \x01(\t\x12+\n\x05shape\x18\x03 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\"\n\x04type\x18\x04 \x01(\x0e\x32\x14.tensorflow.DataType\"\x8e\x01\n\x08Variable\x12\x11\n\tnode_name\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12+\n\x05shape\x18\x03 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\"\n\x04type\x18\x04 \x01(\x0e\x32\x14.tensorflow.DataType\x12\x10\n\x08readonly\x18\x05 \x01(\x08\"\x87\x01\n\x06\x43onfig\x12%\n\x04\x66\x65\x65\x64\x18\x01 \x03(\x0b\x32\x17.tensorflow.tf2xla.Feed\x12\'\n\x05\x66\x65tch\x18\x02 \x03(\x0b\x32\x18.tensorflow.tf2xla.Fetch\x12-\n\x08variable\x18\x03 \x03(\x0b\x32\x1b.tensorflow.tf2xla.VariableB*\n\x15org.tensorflow.tf2xlaB\x0cTf2XlaProtosP\x01\xf8\x01\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.compiler.tf2xla.tf2xla_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\025org.tensorflow.tf2xlaB\014Tf2XlaProtosP\001\370\001\001'
  _TENSORID._serialized_start=147
  _TENSORID._serialized_end=198
  _FEED._serialized_start=201
  _FEED._serialized_end=343
  _FETCH._serialized_start=346
  _FETCH._serialized_end=489
  _VARIABLE._serialized_start=492
  _VARIABLE._serialized_end=634
  _CONFIG._serialized_start=637
  _CONFIG._serialized_end=772
# @@protoc_insertion_point(module_scope)
