# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/compiler/tf2tensorrt/utils/trt_engine_instance.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/compiler/tf2tensorrt/utils/trt_engine_instance.proto',
  package='tensorflow.tensorrt',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n?tensorflow/compiler/tf2tensorrt/utils/trt_engine_instance.proto\x12\x13tensorflow.tensorrt\x1a,tensorflow/core/framework/tensor_shape.proto\"b\n\x11TRTEngineInstance\x12\x32\n\x0cinput_shapes\x18\x01 \x03(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\x19\n\x11serialized_engine\x18\x02 \x01(\x0c\x62\x06proto3')
  ,
  dependencies=[tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2.DESCRIPTOR,])




_TRTENGINEINSTANCE = _descriptor.Descriptor(
  name='TRTEngineInstance',
  full_name='tensorflow.tensorrt.TRTEngineInstance',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input_shapes', full_name='tensorflow.tensorrt.TRTEngineInstance.input_shapes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='serialized_engine', full_name='tensorflow.tensorrt.TRTEngineInstance.serialized_engine', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=134,
  serialized_end=232,
)

_TRTENGINEINSTANCE.fields_by_name['input_shapes'].message_type = tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2._TENSORSHAPEPROTO
DESCRIPTOR.message_types_by_name['TRTEngineInstance'] = _TRTENGINEINSTANCE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TRTEngineInstance = _reflection.GeneratedProtocolMessageType('TRTEngineInstance', (_message.Message,), {
  'DESCRIPTOR' : _TRTENGINEINSTANCE,
  '__module__' : 'tensorflow.compiler.tf2tensorrt.utils.trt_engine_instance_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.tensorrt.TRTEngineInstance)
  })
_sym_db.RegisterMessage(TRTEngineInstance)


# @@protoc_insertion_point(module_scope)
