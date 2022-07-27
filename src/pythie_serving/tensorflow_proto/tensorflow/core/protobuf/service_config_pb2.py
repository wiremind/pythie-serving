# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/service_config.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n-tensorflow/core/protobuf/service_config.proto\x12\x1ctensorflow.data.experimental\"\x9e\x01\n\x10\x44ispatcherConfig\x12\x0c\n\x04port\x18\x01 \x01(\x03\x12\x10\n\x08protocol\x18\x02 \x01(\t\x12\x10\n\x08work_dir\x18\x03 \x01(\t\x12\x1b\n\x13\x66\x61ult_tolerant_mode\x18\x04 \x01(\x08\x12 \n\x18job_gc_check_interval_ms\x18\x05 \x01(\x03\x12\x19\n\x11job_gc_timeout_ms\x18\x06 \x01(\x03\"\x81\x02\n\x0cWorkerConfig\x12\x0c\n\x04port\x18\x01 \x01(\x03\x12\x10\n\x08protocol\x18\x02 \x01(\t\x12\x1a\n\x12\x64ispatcher_address\x18\x03 \x01(\t\x12\x16\n\x0eworker_address\x18\x04 \x01(\t\x12\x1d\n\x15heartbeat_interval_ms\x18\x05 \x01(\x03\x12\x1d\n\x15\x64ispatcher_timeout_ms\x18\x06 \x01(\x03\x12\x1e\n\x16\x64\x61ta_transfer_protocol\x18\x07 \x01(\t\x12\x1d\n\x15\x64\x61ta_transfer_address\x18\x08 \x01(\t\x12 \n\x18shutdown_quiet_period_ms\x18\t \x01(\x03\x42WZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_protob\x06proto3')



_DISPATCHERCONFIG = DESCRIPTOR.message_types_by_name['DispatcherConfig']
_WORKERCONFIG = DESCRIPTOR.message_types_by_name['WorkerConfig']
DispatcherConfig = _reflection.GeneratedProtocolMessageType('DispatcherConfig', (_message.Message,), {
  'DESCRIPTOR' : _DISPATCHERCONFIG,
  '__module__' : 'tensorflow.core.protobuf.service_config_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.DispatcherConfig)
  })
_sym_db.RegisterMessage(DispatcherConfig)

WorkerConfig = _reflection.GeneratedProtocolMessageType('WorkerConfig', (_message.Message,), {
  'DESCRIPTOR' : _WORKERCONFIG,
  '__module__' : 'tensorflow.core.protobuf.service_config_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.experimental.WorkerConfig)
  })
_sym_db.RegisterMessage(WorkerConfig)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto'
  _DISPATCHERCONFIG._serialized_start=80
  _DISPATCHERCONFIG._serialized_end=238
  _WORKERCONFIG._serialized_start=241
  _WORKERCONFIG._serialized_end=498
# @@protoc_insertion_point(module_scope)