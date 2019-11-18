# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/data/experimental/snapshot.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/protobuf/data/experimental/snapshot.proto',
  package='tensorflow.data.experimental',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n9tensorflow/core/protobuf/data/experimental/snapshot.proto\x12\x1ctensorflow.data.experimental\x1a&tensorflow/core/framework/tensor.proto\"9\n\x0eSnapshotRecord\x12\'\n\x06tensor\x18\x01 \x03(\x0b\x32\x17.tensorflow.TensorProto\"l\n\x16SnapshotMetadataRecord\x12\x12\n\ngraph_hash\x18\x01 \x01(\t\x12\x0e\n\x06run_id\x18\x02 \x01(\t\x12\x1a\n\x12\x63reation_timestamp\x18\x03 \x01(\x03\x12\x12\n\tfinalized\x18\xe8\x07 \x01(\x08\x62\x06proto3')
  ,
  dependencies=[tensorflow_dot_core_dot_framework_dot_tensor__pb2.DESCRIPTOR,])




_SNAPSHOTRECORD = _descriptor.Descriptor(
  name='SnapshotRecord',
  full_name='tensorflow.data.experimental.SnapshotRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tensor', full_name='tensorflow.data.experimental.SnapshotRecord.tensor', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=131,
  serialized_end=188,
)


_SNAPSHOTMETADATARECORD = _descriptor.Descriptor(
  name='SnapshotMetadataRecord',
  full_name='tensorflow.data.experimental.SnapshotMetadataRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='graph_hash', full_name='tensorflow.data.experimental.SnapshotMetadataRecord.graph_hash', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='run_id', full_name='tensorflow.data.experimental.SnapshotMetadataRecord.run_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='creation_timestamp', full_name='tensorflow.data.experimental.SnapshotMetadataRecord.creation_timestamp', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='finalized', full_name='tensorflow.data.experimental.SnapshotMetadataRecord.finalized', index=3,
      number=1000, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=190,
  serialized_end=298,
)

_SNAPSHOTRECORD.fields_by_name['tensor'].message_type = tensorflow_dot_core_dot_framework_dot_tensor__pb2._TENSORPROTO
DESCRIPTOR.message_types_by_name['SnapshotRecord'] = _SNAPSHOTRECORD
DESCRIPTOR.message_types_by_name['SnapshotMetadataRecord'] = _SNAPSHOTMETADATARECORD
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SnapshotRecord = _reflection.GeneratedProtocolMessageType('SnapshotRecord', (_message.Message,), {
  'DESCRIPTOR' : _SNAPSHOTRECORD,
  '__module__' : 'tensorflow.core.protobuf.data.experimental.snapshot_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.SnapshotRecord)
  })
_sym_db.RegisterMessage(SnapshotRecord)

SnapshotMetadataRecord = _reflection.GeneratedProtocolMessageType('SnapshotMetadataRecord', (_message.Message,), {
  'DESCRIPTOR' : _SNAPSHOTMETADATARECORD,
  '__module__' : 'tensorflow.core.protobuf.data.experimental.snapshot_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.SnapshotMetadataRecord)
  })
_sym_db.RegisterMessage(SnapshotMetadataRecord)


# @@protoc_insertion_point(module_scope)