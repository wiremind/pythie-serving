# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/dtensor/proto/layout.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n%tensorflow/dtensor/proto/layout.proto\x12\x12tensorflow.dtensor\"+\n\x0cShardingSpec\x12\x15\n\rsharding_spec\x18\x02 \x01(\tJ\x04\x08\x01\x10\x02\"0\n\x12MeshDimensionProto\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04size\x18\x02 \x01(\x03\"{\n\x0bLayoutProto\x12\x38\n\x0esharding_specs\x18\x01 \x03(\x0b\x32 .tensorflow.dtensor.ShardingSpec\x12\x32\n\x0bmesh_config\x18\x02 \x01(\x0b\x32\x1d.tensorflow.dtensor.MeshProto\"\xbe\x01\n\tMeshProto\x12?\n\x0fmesh_dimensions\x18\x01 \x03(\x0b\x32&.tensorflow.dtensor.MeshDimensionProto\x12\x19\n\x11global_device_ids\x18\x02 \x03(\x03\x12\x18\n\x10local_device_ids\x18\x04 \x03(\x03\x12\x15\n\rlocal_devices\x18\x05 \x03(\t\x12\x16\n\x0eglobal_devices\x18\x06 \x03(\t\x12\x0c\n\x04name\x18\x03 \x01(\tb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.dtensor.proto.layout_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SHARDINGSPEC._serialized_start=61
  _SHARDINGSPEC._serialized_end=104
  _MESHDIMENSIONPROTO._serialized_start=106
  _MESHDIMENSIONPROTO._serialized_end=154
  _LAYOUTPROTO._serialized_start=156
  _LAYOUTPROTO._serialized_end=279
  _MESHPROTO._serialized_start=282
  _MESHPROTO._serialized_end=472
# @@protoc_insertion_point(module_scope)
