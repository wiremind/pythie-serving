# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/snapshot.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.framework import types_pb2 as tensorflow_dot_core_dot_framework_dot_types__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\'tensorflow/core/protobuf/snapshot.proto\x12\x1ctensorflow.data.experimental\x1a&tensorflow/core/framework/tensor.proto\x1a,tensorflow/core/framework/tensor_shape.proto\x1a%tensorflow/core/framework/types.proto\"9\n\x0eSnapshotRecord\x12\'\n\x06tensor\x18\x01 \x03(\x0b\x32\x17.tensorflow.TensorProto\"\xb8\x01\n\x16SnapshotMetadataRecord\x12\x12\n\ngraph_hash\x18\x01 \x01(\t\x12\x0e\n\x06run_id\x18\x02 \x01(\t\x12\x1a\n\x12\x63reation_timestamp\x18\x03 \x01(\x03\x12\x0f\n\x07version\x18\x04 \x01(\x03\x12#\n\x05\x64type\x18\x05 \x03(\x0e\x32\x14.tensorflow.DataType\x12\x14\n\x0cnum_elements\x18\x06 \x01(\x03\x12\x12\n\tfinalized\x18\xe8\x07 \x01(\x08\"_\n\x0eTensorMetadata\x12\x32\n\x0ctensor_shape\x18\x02 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\x19\n\x11tensor_size_bytes\x18\x03 \x01(\x03\"_\n\x16SnapshotTensorMetadata\x12\x45\n\x0ftensor_metadata\x18\x01 \x03(\x0b\x32,.tensorflow.data.experimental.TensorMetadataBWZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_protob\x06proto3')



_SNAPSHOTRECORD = DESCRIPTOR.message_types_by_name['SnapshotRecord']
_SNAPSHOTMETADATARECORD = DESCRIPTOR.message_types_by_name['SnapshotMetadataRecord']
_TENSORMETADATA = DESCRIPTOR.message_types_by_name['TensorMetadata']
_SNAPSHOTTENSORMETADATA = DESCRIPTOR.message_types_by_name['SnapshotTensorMetadata']
SnapshotRecord = _reflection.GeneratedProtocolMessageType('SnapshotRecord', (_message.Message,), {
  'DESCRIPTOR' : _SNAPSHOTRECORD,
  '__module__' : 'tensorflow.core.protobuf.snapshot_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.SnapshotRecord)
  })
_sym_db.RegisterMessage(SnapshotRecord)

SnapshotMetadataRecord = _reflection.GeneratedProtocolMessageType('SnapshotMetadataRecord', (_message.Message,), {
  'DESCRIPTOR' : _SNAPSHOTMETADATARECORD,
  '__module__' : 'tensorflow.core.protobuf.snapshot_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.SnapshotMetadataRecord)
  })
_sym_db.RegisterMessage(SnapshotMetadataRecord)

TensorMetadata = _reflection.GeneratedProtocolMessageType('TensorMetadata', (_message.Message,), {
  'DESCRIPTOR' : _TENSORMETADATA,
  '__module__' : 'tensorflow.core.protobuf.snapshot_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.TensorMetadata)
  })
_sym_db.RegisterMessage(TensorMetadata)

SnapshotTensorMetadata = _reflection.GeneratedProtocolMessageType('SnapshotTensorMetadata', (_message.Message,), {
  'DESCRIPTOR' : _SNAPSHOTTENSORMETADATA,
  '__module__' : 'tensorflow.core.protobuf.snapshot_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.SnapshotTensorMetadata)
  })
_sym_db.RegisterMessage(SnapshotTensorMetadata)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto'
  _SNAPSHOTRECORD._serialized_start=198
  _SNAPSHOTRECORD._serialized_end=255
  _SNAPSHOTMETADATARECORD._serialized_start=258
  _SNAPSHOTMETADATARECORD._serialized_end=442
  _TENSORMETADATA._serialized_start=444
  _TENSORMETADATA._serialized_end=539
  _SNAPSHOTTENSORMETADATA._serialized_start=541
  _SNAPSHOTTENSORMETADATA._serialized_end=636
# @@protoc_insertion_point(module_scope)