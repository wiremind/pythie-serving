# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/profiler/protobuf/op_stats.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.profiler.protobuf import diagnostics_pb2 as tensorflow_dot_core_dot_profiler_dot_protobuf_dot_diagnostics__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.profiler.protobuf import kernel_stats_pb2 as tensorflow_dot_core_dot_profiler_dot_protobuf_dot_kernel__stats__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.profiler.protobuf import op_metrics_pb2 as tensorflow_dot_core_dot_profiler_dot_protobuf_dot_op__metrics__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.profiler.protobuf import steps_db_pb2 as tensorflow_dot_core_dot_profiler_dot_protobuf_dot_steps__db__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.profiler.protobuf import tf_function_pb2 as tensorflow_dot_core_dot_profiler_dot_protobuf_dot_tf__function__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n0tensorflow/core/profiler/protobuf/op_stats.proto\x12\x13tensorflow.profiler\x1a\x33tensorflow/core/profiler/protobuf/diagnostics.proto\x1a\x34tensorflow/core/profiler/protobuf/kernel_stats.proto\x1a\x32tensorflow/core/profiler/protobuf/op_metrics.proto\x1a\x30tensorflow/core/profiler/protobuf/steps_db.proto\x1a\x33tensorflow/core/profiler/protobuf/tf_function.proto\"m\n\x07PerfEnv\x12\"\n\x1apeak_tera_flops_per_second\x18\x01 \x01(\x01\x12)\n!peak_hbm_bw_giga_bytes_per_second\x18\x02 \x01(\x01\x12\x13\n\x0bridge_point\x18\x03 \x01(\x01\"z\n\x1cHostIndependentJobInfoResult\x12\x13\n\x0b\x63hange_list\x18\x01 \x01(\x03\x12\x12\n\nbuild_time\x18\x02 \x01(\x03\x12\x14\n\x0c\x62uild_target\x18\x03 \x01(\t\x12\x1b\n\x13profile_duration_ms\x18\x04 \x01(\r\"\x85\x01\n\x1aHostDependentJobInfoResult\x12\x0f\n\x07host_id\x18\x01 \x01(\t\x12\x14\n\x0c\x63ommand_line\x18\x02 \x01(\t\x12\x12\n\nstart_time\x18\x03 \x01(\x03\x12\x13\n\x0b\x62ns_address\x18\x04 \x01(\t\x12\x17\n\x0fprofile_time_ns\x18\x05 \x01(\x04\"s\n\x0eSystemTopology\x12\x13\n\x0bx_dimension\x18\x01 \x01(\x03\x12\x13\n\x0by_dimension\x18\x02 \x01(\x03\x12\x13\n\x0bz_dimension\x18\x03 \x01(\x03\x12\"\n\x1anum_expected_reduced_chips\x18\x04 \x01(\x03\"\xad\x04\n\x0eRunEnvironment\x12\x12\n\nhost_count\x18\x01 \x01(\x05\x12\x12\n\ntask_count\x18\x02 \x01(\x05\x12\x45\n\thostnames\x18\x03 \x03(\x0b\x32\x32.tensorflow.profiler.RunEnvironment.HostnamesEntry\x12\x13\n\x0b\x64\x65vice_type\x18\x04 \x01(\t\x12\x19\n\x11\x64\x65vice_core_count\x18\x05 \x01(\x05\x12\x1b\n\x13per_core_batch_size\x18\x06 \x01(\x05\x12T\n\x19host_independent_job_info\x18\x07 \x01(\x0b\x32\x31.tensorflow.profiler.HostIndependentJobInfoResult\x12P\n\x17host_dependent_job_info\x18\x08 \x03(\x0b\x32/.tensorflow.profiler.HostDependentJobInfoResult\x12\x15\n\rreplica_count\x18\t \x01(\x05\x12\x1d\n\x15num_cores_per_replica\x18\n \x01(\x05\x12\x35\n\x08topology\x18\x0b \x01(\x0b\x32#.tensorflow.profiler.SystemTopology\x12\x18\n\x10host_trace_level\x18\x0c \x01(\r\x1a\x30\n\x0eHostnamesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x08:\x02\x38\x01\"\x90\x01\n\x0b\x43oreDetails\x12\x10\n\x08hostname\x18\x01 \x01(\t\x12\x16\n\x0e\x64\x65vice_ordinal\x18\x02 \x01(\r\x12\x10\n\x08\x63ore_num\x18\x03 \x01(\r\x12\x15\n\rlocal_chip_id\x18\x04 \x01(\r\x12\x16\n\x0eglobal_chip_id\x18\x05 \x01(\r\x12\x16\n\x0eglobal_core_id\x18\x06 \x01(\r\"\xdb\x05\n\x07OpStats\x12<\n\x12host_op_metrics_db\x18\x01 \x01(\x0b\x32 .tensorflow.profiler.OpMetricsDb\x12>\n\x14\x64\x65vice_op_metrics_db\x18\x02 \x01(\x0b\x32 .tensorflow.profiler.OpMetricsDb\x12L\n\"hlo_metrics_db_complete_steps_only\x18\n \x01(\x0b\x32 .tensorflow.profiler.OpMetricsDb\x12.\n\x08perf_env\x18\x03 \x01(\x0b\x32\x1c.tensorflow.profiler.PerfEnv\x12\x38\n\x07step_db\x18\x04 \x01(\x0b\x32\'.tensorflow.profiler.StepDatabaseResult\x12<\n\x0frun_environment\x18\x05 \x01(\x0b\x32#.tensorflow.profiler.RunEnvironment\x12;\n\x0fkernel_stats_db\x18\x06 \x01(\x0b\x32\".tensorflow.profiler.KernelStatsDb\x12\x39\n\x0etf_function_db\x18\x08 \x01(\x0b\x32!.tensorflow.profiler.TfFunctionDb\x12M\n\x12\x63ore_id_to_details\x18\x0b \x03(\x0b\x32\x31.tensorflow.profiler.OpStats.CoreIdToDetailsEntry\x12\x35\n\x0b\x64iagnostics\x18\t \x01(\x0b\x32 .tensorflow.profiler.Diagnostics\x1aX\n\x14\x43oreIdToDetailsEntry\x12\x0b\n\x03key\x18\x01 \x01(\r\x12/\n\x05value\x18\x02 \x01(\x0b\x32 .tensorflow.profiler.CoreDetails:\x02\x38\x01J\x04\x08\x07\x10\x08\x62\x06proto3')



_PERFENV = DESCRIPTOR.message_types_by_name['PerfEnv']
_HOSTINDEPENDENTJOBINFORESULT = DESCRIPTOR.message_types_by_name['HostIndependentJobInfoResult']
_HOSTDEPENDENTJOBINFORESULT = DESCRIPTOR.message_types_by_name['HostDependentJobInfoResult']
_SYSTEMTOPOLOGY = DESCRIPTOR.message_types_by_name['SystemTopology']
_RUNENVIRONMENT = DESCRIPTOR.message_types_by_name['RunEnvironment']
_RUNENVIRONMENT_HOSTNAMESENTRY = _RUNENVIRONMENT.nested_types_by_name['HostnamesEntry']
_COREDETAILS = DESCRIPTOR.message_types_by_name['CoreDetails']
_OPSTATS = DESCRIPTOR.message_types_by_name['OpStats']
_OPSTATS_COREIDTODETAILSENTRY = _OPSTATS.nested_types_by_name['CoreIdToDetailsEntry']
PerfEnv = _reflection.GeneratedProtocolMessageType('PerfEnv', (_message.Message,), {
  'DESCRIPTOR' : _PERFENV,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.PerfEnv)
  })
_sym_db.RegisterMessage(PerfEnv)

HostIndependentJobInfoResult = _reflection.GeneratedProtocolMessageType('HostIndependentJobInfoResult', (_message.Message,), {
  'DESCRIPTOR' : _HOSTINDEPENDENTJOBINFORESULT,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.HostIndependentJobInfoResult)
  })
_sym_db.RegisterMessage(HostIndependentJobInfoResult)

HostDependentJobInfoResult = _reflection.GeneratedProtocolMessageType('HostDependentJobInfoResult', (_message.Message,), {
  'DESCRIPTOR' : _HOSTDEPENDENTJOBINFORESULT,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.HostDependentJobInfoResult)
  })
_sym_db.RegisterMessage(HostDependentJobInfoResult)

SystemTopology = _reflection.GeneratedProtocolMessageType('SystemTopology', (_message.Message,), {
  'DESCRIPTOR' : _SYSTEMTOPOLOGY,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.SystemTopology)
  })
_sym_db.RegisterMessage(SystemTopology)

RunEnvironment = _reflection.GeneratedProtocolMessageType('RunEnvironment', (_message.Message,), {

  'HostnamesEntry' : _reflection.GeneratedProtocolMessageType('HostnamesEntry', (_message.Message,), {
    'DESCRIPTOR' : _RUNENVIRONMENT_HOSTNAMESENTRY,
    '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.profiler.RunEnvironment.HostnamesEntry)
    })
  ,
  'DESCRIPTOR' : _RUNENVIRONMENT,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.RunEnvironment)
  })
_sym_db.RegisterMessage(RunEnvironment)
_sym_db.RegisterMessage(RunEnvironment.HostnamesEntry)

CoreDetails = _reflection.GeneratedProtocolMessageType('CoreDetails', (_message.Message,), {
  'DESCRIPTOR' : _COREDETAILS,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.CoreDetails)
  })
_sym_db.RegisterMessage(CoreDetails)

OpStats = _reflection.GeneratedProtocolMessageType('OpStats', (_message.Message,), {

  'CoreIdToDetailsEntry' : _reflection.GeneratedProtocolMessageType('CoreIdToDetailsEntry', (_message.Message,), {
    'DESCRIPTOR' : _OPSTATS_COREIDTODETAILSENTRY,
    '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.profiler.OpStats.CoreIdToDetailsEntry)
    })
  ,
  'DESCRIPTOR' : _OPSTATS,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_stats_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.OpStats)
  })
_sym_db.RegisterMessage(OpStats)
_sym_db.RegisterMessage(OpStats.CoreIdToDetailsEntry)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _RUNENVIRONMENT_HOSTNAMESENTRY._options = None
  _RUNENVIRONMENT_HOSTNAMESENTRY._serialized_options = b'8\001'
  _OPSTATS_COREIDTODETAILSENTRY._options = None
  _OPSTATS_COREIDTODETAILSENTRY._serialized_options = b'8\001'
  _PERFENV._serialized_start=335
  _PERFENV._serialized_end=444
  _HOSTINDEPENDENTJOBINFORESULT._serialized_start=446
  _HOSTINDEPENDENTJOBINFORESULT._serialized_end=568
  _HOSTDEPENDENTJOBINFORESULT._serialized_start=571
  _HOSTDEPENDENTJOBINFORESULT._serialized_end=704
  _SYSTEMTOPOLOGY._serialized_start=706
  _SYSTEMTOPOLOGY._serialized_end=821
  _RUNENVIRONMENT._serialized_start=824
  _RUNENVIRONMENT._serialized_end=1381
  _RUNENVIRONMENT_HOSTNAMESENTRY._serialized_start=1333
  _RUNENVIRONMENT_HOSTNAMESENTRY._serialized_end=1381
  _COREDETAILS._serialized_start=1384
  _COREDETAILS._serialized_end=1528
  _OPSTATS._serialized_start=1531
  _OPSTATS._serialized_end=2262
  _OPSTATS_COREIDTODETAILSENTRY._serialized_start=2168
  _OPSTATS_COREIDTODETAILSENTRY._serialized_end=2256
# @@protoc_insertion_point(module_scope)
