# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/saver.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$tensorflow/core/protobuf/saver.proto\x12\ntensorflow\"\x9e\x02\n\x08SaverDef\x12\x1c\n\x14\x66ilename_tensor_name\x18\x01 \x01(\t\x12\x18\n\x10save_tensor_name\x18\x02 \x01(\t\x12\x17\n\x0frestore_op_name\x18\x03 \x01(\t\x12\x13\n\x0bmax_to_keep\x18\x04 \x01(\x05\x12\x0f\n\x07sharded\x18\x05 \x01(\x08\x12%\n\x1dkeep_checkpoint_every_n_hours\x18\x06 \x01(\x02\x12=\n\x07version\x18\x07 \x01(\x0e\x32,.tensorflow.SaverDef.CheckpointFormatVersion\"5\n\x17\x43heckpointFormatVersion\x12\n\n\x06LEGACY\x10\x00\x12\x06\n\x02V1\x10\x01\x12\x06\n\x02V2\x10\x02\x42~\n\x13org.tensorflow.utilB\x0bSaverProtosP\x01ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\xf8\x01\x01\x62\x06proto3')



_SAVERDEF = DESCRIPTOR.message_types_by_name['SaverDef']
_SAVERDEF_CHECKPOINTFORMATVERSION = _SAVERDEF.enum_types_by_name['CheckpointFormatVersion']
SaverDef = _reflection.GeneratedProtocolMessageType('SaverDef', (_message.Message,), {
  'DESCRIPTOR' : _SAVERDEF,
  '__module__' : 'tensorflow.core.protobuf.saver_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.SaverDef)
  })
_sym_db.RegisterMessage(SaverDef)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\023org.tensorflow.utilB\013SaverProtosP\001ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\370\001\001'
  _SAVERDEF._serialized_start=53
  _SAVERDEF._serialized_end=339
  _SAVERDEF_CHECKPOINTFORMATVERSION._serialized_start=286
  _SAVERDEF_CHECKPOINTFORMATVERSION._serialized_end=339
# @@protoc_insertion_point(module_scope)
