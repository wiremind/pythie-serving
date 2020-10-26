# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/profiler/protobuf/op_metrics.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/profiler/protobuf/op_metrics.proto',
  package='tensorflow.profiler',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n2tensorflow/core/profiler/protobuf/op_metrics.proto\x12\x13tensorflow.profiler\"\xc3\x01\n\x0eLayoutAnalysis\x12\x41\n\ndimensions\x18\x01 \x03(\x0b\x32-.tensorflow.profiler.LayoutAnalysis.Dimension\x1an\n\tDimension\x12\x0c\n\x04size\x18\x01 \x01(\x05\x12\x11\n\talignment\x18\x02 \x01(\x05\x12@\n\tsemantics\x18\x03 \x01(\x0e\x32-.tensorflow.profiler.LayoutDimensionSemantics\"\xa7\x05\n\tOpMetrics\x12\x15\n\rhlo_module_id\x18\r \x01(\x04\x12\x0c\n\x04name\x18\x06 \x01(\t\x12\x10\n\x08\x63\x61tegory\x18\x0b \x01(\t\x12\x12\n\nprovenance\x18\x0c \x01(\t\x12\x10\n\x08is_eager\x18\x12 \x01(\x08\x12\x13\n\x0boccurrences\x18\x03 \x01(\r\x12\x0f\n\x07time_ps\x18\x07 \x01(\x04\x12\x13\n\x0bmin_time_ps\x18\x11 \x01(\x04\x12\x14\n\x0cself_time_ps\x18\x01 \x01(\x04\x12\r\n\x05\x66lops\x18\x02 \x01(\x04\x12\x16\n\x0e\x62ytes_accessed\x18\x05 \x01(\x04\x12P\n\x19memory_accessed_breakdown\x18\x13 \x03(\x0b\x32-.tensorflow.profiler.OpMetrics.MemoryAccessed\x12\x14\n\x0c\x64ma_stall_ps\x18\n \x01(\x04\x12\x33\n\x06layout\x18\x0e \x01(\x0b\x32#.tensorflow.profiler.LayoutAnalysis\x12\x19\n\x11\x64\x65\x64uplicated_name\x18\x0f \x01(\t\x12\x32\n\x08\x63hildren\x18\x10 \x01(\x0b\x32 .tensorflow.profiler.OpMetricsDb\x1a\xc6\x01\n\x0eMemoryAccessed\x12S\n\x0eoperation_type\x18\x01 \x01(\x0e\x32;.tensorflow.profiler.OpMetrics.MemoryAccessed.OperationType\x12\x14\n\x0cmemory_space\x18\x02 \x01(\x04\x12\x16\n\x0e\x62ytes_accessed\x18\x03 \x01(\x04\"1\n\rOperationType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x08\n\x04READ\x10\x01\x12\t\n\x05WRITE\x10\x02J\x04\x08\x04\x10\x05J\x04\x08\x08\x10\tJ\x04\x08\t\x10\n\"D\n\x0ePrecisionStats\x12\x18\n\x10\x63ompute_16bit_ps\x18\x01 \x01(\x04\x12\x18\n\x10\x63ompute_32bit_ps\x18\x02 \x01(\x04\"\xbc\x02\n\x0bOpMetricsDb\x12\x32\n\nmetrics_db\x18\n \x03(\x0b\x32\x1e.tensorflow.profiler.OpMetrics\x12)\n!total_host_infeed_enq_duration_ps\x18\x02 \x01(\x04\x12\x35\n-total_host_infeed_enq_start_timestamp_ps_diff\x18\x03 \x01(\x04\x12\x15\n\rtotal_time_ps\x18\x0b \x01(\x04\x12\x18\n\x10total_op_time_ps\x18\x0c \x01(\x04\x12<\n\x0fprecision_stats\x18\r \x01(\x0b\x32#.tensorflow.profiler.PrecisionStatsJ\x04\x08\x01\x10\x02J\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x06\x10\x07J\x04\x08\x07\x10\x08J\x04\x08\x08\x10\tJ\x04\x08\t\x10\n*V\n\x18LayoutDimensionSemantics\x12\x15\n\x11UNKNOWN_SEMANTICS\x10\x00\x12\x0b\n\x07\x46\x45\x41TURE\x10\x01\x12\t\n\x05\x42\x41TCH\x10\x02\x12\x0b\n\x07SPATIAL\x10\x03\x62\x06proto3'
)

_LAYOUTDIMENSIONSEMANTICS = _descriptor.EnumDescriptor(
  name='LayoutDimensionSemantics',
  full_name='tensorflow.profiler.LayoutDimensionSemantics',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN_SEMANTICS', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FEATURE', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BATCH', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SPATIAL', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1344,
  serialized_end=1430,
)
_sym_db.RegisterEnumDescriptor(_LAYOUTDIMENSIONSEMANTICS)

LayoutDimensionSemantics = enum_type_wrapper.EnumTypeWrapper(_LAYOUTDIMENSIONSEMANTICS)
UNKNOWN_SEMANTICS = 0
FEATURE = 1
BATCH = 2
SPATIAL = 3


_OPMETRICS_MEMORYACCESSED_OPERATIONTYPE = _descriptor.EnumDescriptor(
  name='OperationType',
  full_name='tensorflow.profiler.OpMetrics.MemoryAccessed.OperationType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='READ', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='WRITE', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=886,
  serialized_end=935,
)
_sym_db.RegisterEnumDescriptor(_OPMETRICS_MEMORYACCESSED_OPERATIONTYPE)


_LAYOUTANALYSIS_DIMENSION = _descriptor.Descriptor(
  name='Dimension',
  full_name='tensorflow.profiler.LayoutAnalysis.Dimension',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='size', full_name='tensorflow.profiler.LayoutAnalysis.Dimension.size', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='alignment', full_name='tensorflow.profiler.LayoutAnalysis.Dimension.alignment', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='semantics', full_name='tensorflow.profiler.LayoutAnalysis.Dimension.semantics', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=161,
  serialized_end=271,
)

_LAYOUTANALYSIS = _descriptor.Descriptor(
  name='LayoutAnalysis',
  full_name='tensorflow.profiler.LayoutAnalysis',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dimensions', full_name='tensorflow.profiler.LayoutAnalysis.dimensions', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_LAYOUTANALYSIS_DIMENSION, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=76,
  serialized_end=271,
)


_OPMETRICS_MEMORYACCESSED = _descriptor.Descriptor(
  name='MemoryAccessed',
  full_name='tensorflow.profiler.OpMetrics.MemoryAccessed',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='operation_type', full_name='tensorflow.profiler.OpMetrics.MemoryAccessed.operation_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='memory_space', full_name='tensorflow.profiler.OpMetrics.MemoryAccessed.memory_space', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bytes_accessed', full_name='tensorflow.profiler.OpMetrics.MemoryAccessed.bytes_accessed', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _OPMETRICS_MEMORYACCESSED_OPERATIONTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=737,
  serialized_end=935,
)

_OPMETRICS = _descriptor.Descriptor(
  name='OpMetrics',
  full_name='tensorflow.profiler.OpMetrics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='hlo_module_id', full_name='tensorflow.profiler.OpMetrics.hlo_module_id', index=0,
      number=13, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorflow.profiler.OpMetrics.name', index=1,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='category', full_name='tensorflow.profiler.OpMetrics.category', index=2,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='provenance', full_name='tensorflow.profiler.OpMetrics.provenance', index=3,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='is_eager', full_name='tensorflow.profiler.OpMetrics.is_eager', index=4,
      number=18, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='occurrences', full_name='tensorflow.profiler.OpMetrics.occurrences', index=5,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='time_ps', full_name='tensorflow.profiler.OpMetrics.time_ps', index=6,
      number=7, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='min_time_ps', full_name='tensorflow.profiler.OpMetrics.min_time_ps', index=7,
      number=17, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='self_time_ps', full_name='tensorflow.profiler.OpMetrics.self_time_ps', index=8,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='flops', full_name='tensorflow.profiler.OpMetrics.flops', index=9,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bytes_accessed', full_name='tensorflow.profiler.OpMetrics.bytes_accessed', index=10,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='memory_accessed_breakdown', full_name='tensorflow.profiler.OpMetrics.memory_accessed_breakdown', index=11,
      number=19, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dma_stall_ps', full_name='tensorflow.profiler.OpMetrics.dma_stall_ps', index=12,
      number=10, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='layout', full_name='tensorflow.profiler.OpMetrics.layout', index=13,
      number=14, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='deduplicated_name', full_name='tensorflow.profiler.OpMetrics.deduplicated_name', index=14,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='children', full_name='tensorflow.profiler.OpMetrics.children', index=15,
      number=16, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_OPMETRICS_MEMORYACCESSED, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=274,
  serialized_end=953,
)


_PRECISIONSTATS = _descriptor.Descriptor(
  name='PrecisionStats',
  full_name='tensorflow.profiler.PrecisionStats',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='compute_16bit_ps', full_name='tensorflow.profiler.PrecisionStats.compute_16bit_ps', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='compute_32bit_ps', full_name='tensorflow.profiler.PrecisionStats.compute_32bit_ps', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=955,
  serialized_end=1023,
)


_OPMETRICSDB = _descriptor.Descriptor(
  name='OpMetricsDb',
  full_name='tensorflow.profiler.OpMetricsDb',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metrics_db', full_name='tensorflow.profiler.OpMetricsDb.metrics_db', index=0,
      number=10, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_host_infeed_enq_duration_ps', full_name='tensorflow.profiler.OpMetricsDb.total_host_infeed_enq_duration_ps', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_host_infeed_enq_start_timestamp_ps_diff', full_name='tensorflow.profiler.OpMetricsDb.total_host_infeed_enq_start_timestamp_ps_diff', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_time_ps', full_name='tensorflow.profiler.OpMetricsDb.total_time_ps', index=3,
      number=11, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_op_time_ps', full_name='tensorflow.profiler.OpMetricsDb.total_op_time_ps', index=4,
      number=12, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='precision_stats', full_name='tensorflow.profiler.OpMetricsDb.precision_stats', index=5,
      number=13, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=1026,
  serialized_end=1342,
)

_LAYOUTANALYSIS_DIMENSION.fields_by_name['semantics'].enum_type = _LAYOUTDIMENSIONSEMANTICS
_LAYOUTANALYSIS_DIMENSION.containing_type = _LAYOUTANALYSIS
_LAYOUTANALYSIS.fields_by_name['dimensions'].message_type = _LAYOUTANALYSIS_DIMENSION
_OPMETRICS_MEMORYACCESSED.fields_by_name['operation_type'].enum_type = _OPMETRICS_MEMORYACCESSED_OPERATIONTYPE
_OPMETRICS_MEMORYACCESSED.containing_type = _OPMETRICS
_OPMETRICS_MEMORYACCESSED_OPERATIONTYPE.containing_type = _OPMETRICS_MEMORYACCESSED
_OPMETRICS.fields_by_name['memory_accessed_breakdown'].message_type = _OPMETRICS_MEMORYACCESSED
_OPMETRICS.fields_by_name['layout'].message_type = _LAYOUTANALYSIS
_OPMETRICS.fields_by_name['children'].message_type = _OPMETRICSDB
_OPMETRICSDB.fields_by_name['metrics_db'].message_type = _OPMETRICS
_OPMETRICSDB.fields_by_name['precision_stats'].message_type = _PRECISIONSTATS
DESCRIPTOR.message_types_by_name['LayoutAnalysis'] = _LAYOUTANALYSIS
DESCRIPTOR.message_types_by_name['OpMetrics'] = _OPMETRICS
DESCRIPTOR.message_types_by_name['PrecisionStats'] = _PRECISIONSTATS
DESCRIPTOR.message_types_by_name['OpMetricsDb'] = _OPMETRICSDB
DESCRIPTOR.enum_types_by_name['LayoutDimensionSemantics'] = _LAYOUTDIMENSIONSEMANTICS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

LayoutAnalysis = _reflection.GeneratedProtocolMessageType('LayoutAnalysis', (_message.Message,), {

  'Dimension' : _reflection.GeneratedProtocolMessageType('Dimension', (_message.Message,), {
    'DESCRIPTOR' : _LAYOUTANALYSIS_DIMENSION,
    '__module__' : 'tensorflow.core.profiler.protobuf.op_metrics_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.profiler.LayoutAnalysis.Dimension)
    })
  ,
  'DESCRIPTOR' : _LAYOUTANALYSIS,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_metrics_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.LayoutAnalysis)
  })
_sym_db.RegisterMessage(LayoutAnalysis)
_sym_db.RegisterMessage(LayoutAnalysis.Dimension)

OpMetrics = _reflection.GeneratedProtocolMessageType('OpMetrics', (_message.Message,), {

  'MemoryAccessed' : _reflection.GeneratedProtocolMessageType('MemoryAccessed', (_message.Message,), {
    'DESCRIPTOR' : _OPMETRICS_MEMORYACCESSED,
    '__module__' : 'tensorflow.core.profiler.protobuf.op_metrics_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.profiler.OpMetrics.MemoryAccessed)
    })
  ,
  'DESCRIPTOR' : _OPMETRICS,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_metrics_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.OpMetrics)
  })
_sym_db.RegisterMessage(OpMetrics)
_sym_db.RegisterMessage(OpMetrics.MemoryAccessed)

PrecisionStats = _reflection.GeneratedProtocolMessageType('PrecisionStats', (_message.Message,), {
  'DESCRIPTOR' : _PRECISIONSTATS,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_metrics_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.PrecisionStats)
  })
_sym_db.RegisterMessage(PrecisionStats)

OpMetricsDb = _reflection.GeneratedProtocolMessageType('OpMetricsDb', (_message.Message,), {
  'DESCRIPTOR' : _OPMETRICSDB,
  '__module__' : 'tensorflow.core.profiler.protobuf.op_metrics_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.profiler.OpMetricsDb)
  })
_sym_db.RegisterMessage(OpMetricsDb)


# @@protoc_insertion_point(module_scope)