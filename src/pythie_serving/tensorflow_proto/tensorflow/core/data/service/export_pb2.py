# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/data/service/export.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.data.service import common_pb2 as tensorflow_dot_core_dot_data_dot_service_dot_common__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.protobuf import data_service_pb2 as tensorflow_dot_core_dot_protobuf_dot_data__service__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.protobuf import service_config_pb2 as tensorflow_dot_core_dot_protobuf_dot_service__config__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)tensorflow/core/data/service/export.proto\x12\x0ftensorflow.data\x1a)tensorflow/core/data/service/common.proto\x1a+tensorflow/core/protobuf/data_service.proto\x1a-tensorflow/core/protobuf/service_config.proto\"\xc9\x03\n\x15\x44ispatcherStateExport\x12I\n\x11\x64ispatcher_config\x18\x01 \x01(\x0b\x32..tensorflow.data.experimental.DispatcherConfig\x12\x18\n\x10worker_addresses\x18\x02 \x03(\t\x12\x44\n\niterations\x18\x03 \x03(\x0b\x32\x30.tensorflow.data.DispatcherStateExport.Iteration\x1a\x84\x02\n\tIteration\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x14\n\x0citeration_id\x18\x02 \x01(\x03\x12\x37\n\riteration_key\x18\x03 \x01(\x0b\x32 .tensorflow.data.IterationKeyDef\x12;\n\x0fprocessing_mode\x18\x04 \x01(\x0b\x32\".tensorflow.data.ProcessingModeDef\x12\x15\n\rnum_consumers\x18\x06 \x01(\x03\x12\x13\n\x0bnum_clients\x18\x08 \x01(\x03\x12\x10\n\x08\x66inished\x18\n \x01(\x08\x12\x19\n\x11garbage_collected\x18\x0b \x01(\x08\"\xb4\x01\n\x11WorkerStateExport\x12\x41\n\rworker_config\x18\x01 \x01(\x0b\x32*.tensorflow.data.experimental.WorkerConfig\x12\'\n\x05tasks\x18\x02 \x03(\x0b\x32\x18.tensorflow.data.TaskDef\x12\x19\n\x11\x66inished_task_ids\x18\x03 \x03(\x03\x12\x18\n\x10\x64\x65leted_task_ids\x18\x04 \x03(\x03\"\x9d\x01\n\x11ServerStateExport\x12G\n\x17\x64ispatcher_state_export\x18\x01 \x01(\x0b\x32&.tensorflow.data.DispatcherStateExport\x12?\n\x13worker_state_export\x18\x02 \x01(\x0b\x32\".tensorflow.data.WorkerStateExportb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.core.data.service.export_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _DISPATCHERSTATEEXPORT._serialized_start=198
  _DISPATCHERSTATEEXPORT._serialized_end=655
  _DISPATCHERSTATEEXPORT_ITERATION._serialized_start=395
  _DISPATCHERSTATEEXPORT_ITERATION._serialized_end=655
  _WORKERSTATEEXPORT._serialized_start=658
  _WORKERSTATEEXPORT._serialized_end=838
  _SERVERSTATEEXPORT._serialized_start=841
  _SERVERSTATEEXPORT._serialized_end=998
# @@protoc_insertion_point(module_scope)
