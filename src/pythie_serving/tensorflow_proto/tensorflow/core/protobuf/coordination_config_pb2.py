# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/coordination_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n2tensorflow/core/protobuf/coordination_config.proto\x12\ntensorflow\"1\n\x0e\x43oordinatedJob\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tnum_tasks\x18\x02 \x01(\x05\"\xdd\x02\n\x19\x43oordinationServiceConfig\x12\x14\n\x0cservice_type\x18\x01 \x01(\t\x12\x16\n\x0eservice_leader\x18\x02 \x01(\t\x12\x1b\n\x13\x65nable_health_check\x18\x03 \x01(\x08\x12&\n\x1e\x63luster_register_timeout_in_ms\x18\x04 \x01(\x03\x12\x1f\n\x17heartbeat_timeout_in_ms\x18\x05 \x01(\x03\x12\x38\n\x14\x63oordinated_job_list\x18\n \x03(\x0b\x32\x1a.tensorflow.CoordinatedJob\x12&\n\x1eshutdown_barrier_timeout_in_ms\x18\x07 \x01(\x03\x12*\n\"agent_destruction_without_shutdown\x18\x08 \x01(\x08\x12\x18\n\x10recoverable_jobs\x18\t \x03(\tJ\x04\x08\x06\x10\x07\x42WZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_protob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.core.protobuf.coordination_config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto'
  _COORDINATEDJOB._serialized_start=66
  _COORDINATEDJOB._serialized_end=115
  _COORDINATIONSERVICECONFIG._serialized_start=118
  _COORDINATIONSERVICECONFIG._serialized_end=467
# @@protoc_insertion_point(module_scope)
