# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/profiler/profiler_service_monitor_result.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n>tensorflow/core/profiler/profiler_service_monitor_result.proto\x12\ntensorflow\"\xad\x03\n\x1cProfilerServiceMonitorResult\x12L\n\rresponse_type\x18\x01 \x01(\x0e\x32\x35.tensorflow.ProfilerServiceMonitorResult.ResponseType\x12 \n\x18\x64\x65vice_idle_time_percent\x18\x02 \x01(\x01\x12\'\n\x1fmatrix_unit_utilization_percent\x18\x03 \x01(\x01\x12\x18\n\x10step_time_ms_avg\x18\x04 \x01(\x01\x12\x18\n\x10step_time_ms_min\x18\x05 \x01(\x01\x12\x18\n\x10step_time_ms_max\x18\x06 \x01(\x01\x12\x1a\n\x12infeed_percent_avg\x18\x07 \x01(\x01\x12\x1a\n\x12infeed_percent_min\x18\x08 \x01(\x01\x12\x1a\n\x12infeed_percent_max\x18\t \x01(\x01\"R\n\x0cResponseType\x12\x10\n\x0c\x45MPTY_RESULT\x10\x00\x12\r\n\tUTIL_ONLY\x10\x01\x12\r\n\tUTIL_IDLE\x10\x02\x12\x12\n\x0eUTIL_IDLE_STEP\x10\x03\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.core.profiler.profiler_service_monitor_result_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PROFILERSERVICEMONITORRESULT._serialized_start=79
  _PROFILERSERVICEMONITORRESULT._serialized_end=508
  _PROFILERSERVICEMONITORRESULT_RESPONSETYPE._serialized_start=426
  _PROFILERSERVICEMONITORRESULT_RESPONSETYPE._serialized_end=508
# @@protoc_insertion_point(module_scope)
