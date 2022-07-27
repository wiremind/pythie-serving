# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/dataset_options.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n/tensorflow/core/framework/dataset_options.proto\x12\x0ftensorflow.data\"\x7f\n\x11\x44istributeOptions\x12;\n\x11\x61uto_shard_policy\x18\x01 \x01(\x0e\x32 .tensorflow.data.AutoShardPolicy\x12\x15\n\x0bnum_devices\x18\x02 \x01(\x05H\x00\x42\x16\n\x14optional_num_devices\"\xb8\x06\n\x13OptimizationOptions\x12%\n\x1b\x61pply_default_optimizations\x18\x01 \x01(\x08H\x00\x12\x12\n\x08\x61utotune\x18\x02 \x01(\x08H\x01\x12\x1a\n\x10\x61utotune_buffers\x18\x03 \x01(\x08H\x02\x12\x1d\n\x13\x61utotune_cpu_budget\x18\x04 \x01(\x05H\x03\x12\x1d\n\x13\x61utotune_ram_budget\x18\x05 \x01(\x03H\x04\x12\x17\n\rfilter_fusion\x18\x06 \x01(\x08H\x05\x12\x1e\n\x14map_and_batch_fusion\x18\t \x01(\x08H\x06\x12\x1f\n\x15map_and_filter_fusion\x18\n \x01(\x08H\x07\x12\x14\n\nmap_fusion\x18\x0b \x01(\x08H\x08\x12\x1d\n\x13map_parallelization\x18\x0c \x01(\x08H\t\x12\x1a\n\x10noop_elimination\x18\x0e \x01(\x08H\n\x12\x18\n\x0eparallel_batch\x18\x0f \x01(\x08H\x0b\x12#\n\x19shuffle_and_repeat_fusion\x18\x11 \x01(\x08H\x0c\x42&\n$optional_apply_default_optimizationsB\x13\n\x11optional_autotuneB\x1b\n\x19optional_autotune_buffersB\x1e\n\x1coptional_autotune_cpu_budgetB\x1e\n\x1coptional_autotune_ram_budgetB\x18\n\x16optional_filter_fusionB\x1f\n\x1doptional_map_and_batch_fusionB \n\x1eoptional_map_and_filter_fusionB\x15\n\x13optional_map_fusionB\x1e\n\x1coptional_map_parallelizationB\x1b\n\x19optional_noop_eliminationB\x19\n\x17optional_parallel_batchB$\n\"optional_shuffle_and_repeat_fusionJ\x04\x08\x07\x10\x08J\x04\x08\x08\x10\tJ\x04\x08\r\x10\x0eJ\x04\x08\x10\x10\x11\"\xa2\x01\n\x10ThreadingOptions\x12\"\n\x18max_intra_op_parallelism\x18\x01 \x01(\x05H\x00\x12!\n\x17private_threadpool_size\x18\x02 \x01(\x05H\x01\x42#\n!optional_max_intra_op_parallelismB\"\n optional_private_threadpool_size\"\x8a\x03\n\x07Options\x12\x17\n\rdeterministic\x18\x01 \x01(\x08H\x00\x12>\n\x12\x64istribute_options\x18\x02 \x01(\x0b\x32\".tensorflow.data.DistributeOptions\x12\x42\n\x14optimization_options\x18\x03 \x01(\x0b\x32$.tensorflow.data.OptimizationOptions\x12\x0f\n\x05slack\x18\x04 \x01(\x08H\x01\x12<\n\x11threading_options\x18\x05 \x01(\x0b\x32!.tensorflow.data.ThreadingOptions\x12\x45\n\x15\x65xternal_state_policy\x18\x06 \x01(\x0e\x32$.tensorflow.data.ExternalStatePolicyH\x02\x42\x18\n\x16optional_deterministicB\x10\n\x0eoptional_slackB \n\x1eoptional_external_state_policy*K\n\x0f\x41utoShardPolicy\x12\x08\n\x04\x41UTO\x10\x00\x12\x08\n\x04\x46ILE\x10\x01\x12\x08\n\x04\x44\x41TA\x10\x02\x12\x08\n\x04HINT\x10\x03\x12\x10\n\x03OFF\x10\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01*J\n\x13\x45xternalStatePolicy\x12\x0f\n\x0bPOLICY_WARN\x10\x00\x12\x11\n\rPOLICY_IGNORE\x10\x01\x12\x0f\n\x0bPOLICY_FAIL\x10\x02\x42XZVgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/dataset_options_go_protob\x06proto3')

_AUTOSHARDPOLICY = DESCRIPTOR.enum_types_by_name['AutoShardPolicy']
AutoShardPolicy = enum_type_wrapper.EnumTypeWrapper(_AUTOSHARDPOLICY)
_EXTERNALSTATEPOLICY = DESCRIPTOR.enum_types_by_name['ExternalStatePolicy']
ExternalStatePolicy = enum_type_wrapper.EnumTypeWrapper(_EXTERNALSTATEPOLICY)
AUTO = 0
FILE = 1
DATA = 2
HINT = 3
OFF = -1
POLICY_WARN = 0
POLICY_IGNORE = 1
POLICY_FAIL = 2


_DISTRIBUTEOPTIONS = DESCRIPTOR.message_types_by_name['DistributeOptions']
_OPTIMIZATIONOPTIONS = DESCRIPTOR.message_types_by_name['OptimizationOptions']
_THREADINGOPTIONS = DESCRIPTOR.message_types_by_name['ThreadingOptions']
_OPTIONS = DESCRIPTOR.message_types_by_name['Options']
DistributeOptions = _reflection.GeneratedProtocolMessageType('DistributeOptions', (_message.Message,), {
  'DESCRIPTOR' : _DISTRIBUTEOPTIONS,
  '__module__' : 'tensorflow.core.framework.dataset_options_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.DistributeOptions)
  })
_sym_db.RegisterMessage(DistributeOptions)

OptimizationOptions = _reflection.GeneratedProtocolMessageType('OptimizationOptions', (_message.Message,), {
  'DESCRIPTOR' : _OPTIMIZATIONOPTIONS,
  '__module__' : 'tensorflow.core.framework.dataset_options_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.OptimizationOptions)
  })
_sym_db.RegisterMessage(OptimizationOptions)

ThreadingOptions = _reflection.GeneratedProtocolMessageType('ThreadingOptions', (_message.Message,), {
  'DESCRIPTOR' : _THREADINGOPTIONS,
  '__module__' : 'tensorflow.core.framework.dataset_options_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.ThreadingOptions)
  })
_sym_db.RegisterMessage(ThreadingOptions)

Options = _reflection.GeneratedProtocolMessageType('Options', (_message.Message,), {
  'DESCRIPTOR' : _OPTIONS,
  '__module__' : 'tensorflow.core.framework.dataset_options_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.data.Options)
  })
_sym_db.RegisterMessage(Options)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZVgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/dataset_options_go_proto'
  _AUTOSHARDPOLICY._serialized_start=1586
  _AUTOSHARDPOLICY._serialized_end=1661
  _EXTERNALSTATEPOLICY._serialized_start=1663
  _EXTERNALSTATEPOLICY._serialized_end=1737
  _DISTRIBUTEOPTIONS._serialized_start=68
  _DISTRIBUTEOPTIONS._serialized_end=195
  _OPTIMIZATIONOPTIONS._serialized_start=198
  _OPTIMIZATIONOPTIONS._serialized_end=1022
  _THREADINGOPTIONS._serialized_start=1025
  _THREADINGOPTIONS._serialized_end=1187
  _OPTIONS._serialized_start=1190
  _OPTIONS._serialized_end=1584
# @@protoc_insertion_point(module_scope)