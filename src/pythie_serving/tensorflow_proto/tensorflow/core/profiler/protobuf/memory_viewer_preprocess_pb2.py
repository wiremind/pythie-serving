# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/profiler/protobuf/memory_viewer_preprocess.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n@tensorflow/core/profiler/protobuf/memory_viewer_preprocess.proto\x12\x13tensorflow.profiler\"\x8a\x02\n\nHeapObject\x12\x12\n\x08numbered\x18\x01 \x01(\x05H\x00\x12\x0f\n\x05named\x18\x02 \x01(\tH\x00\x12\r\n\x05label\x18\x03 \x01(\t\x12\x19\n\x11logical_buffer_id\x18\x04 \x01(\x05\x12\x1f\n\x17logical_buffer_size_mib\x18\x05 \x01(\x01\x12\x1a\n\x12unpadded_shape_mib\x18\x06 \x01(\x01\x12\x18\n\x10instruction_name\x18\x07 \x01(\t\x12\x14\n\x0cshape_string\x18\x08 \x01(\t\x12\x12\n\ntf_op_name\x18\t \x01(\t\x12\x12\n\ngroup_name\x18\n \x01(\t\x12\x0f\n\x07op_code\x18\x0b \x01(\tB\x07\n\x05\x63olor\"*\n\nBufferSpan\x12\r\n\x05start\x18\x01 \x01(\x05\x12\r\n\x05limit\x18\x02 \x01(\x05\"c\n\rLogicalBuffer\x12\n\n\x02id\x18\x01 \x01(\x03\x12\r\n\x05shape\x18\x02 \x01(\t\x12\x10\n\x08size_mib\x18\x03 \x01(\x01\x12\x10\n\x08hlo_name\x18\x04 \x01(\t\x12\x13\n\x0bshape_index\x18\x05 \x03(\x03\"\x97\x01\n\x10\x42ufferAllocation\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x10\n\x08size_mib\x18\x02 \x01(\x01\x12\x12\n\nattributes\x18\x03 \x03(\t\x12;\n\x0flogical_buffers\x18\x04 \x03(\x0b\x32\".tensorflow.profiler.LogicalBuffer\x12\x14\n\x0c\x63ommon_shape\x18\x05 \x01(\t\"\xd6\x05\n\x10PreprocessResult\x12\x12\n\nheap_sizes\x18\x01 \x03(\x01\x12\x1b\n\x13unpadded_heap_sizes\x18\x02 \x03(\x01\x12\x31\n\x08max_heap\x18\x03 \x03(\x0b\x32\x1f.tensorflow.profiler.HeapObject\x12\x39\n\x10max_heap_by_size\x18\x04 \x03(\x0b\x32\x1f.tensorflow.profiler.HeapObject\x12[\n\x14logical_buffer_spans\x18\x05 \x03(\x0b\x32=.tensorflow.profiler.PreprocessResult.LogicalBufferSpansEntry\x12\x1b\n\x13max_heap_to_by_size\x18\x06 \x03(\x05\x12\x1b\n\x13\x62y_size_to_max_heap\x18\x07 \x03(\x05\x12\x13\n\x0bmodule_name\x18\x08 \x01(\t\x12\x1e\n\x16\x65ntry_computation_name\x18\t \x01(\t\x12\x15\n\rpeak_heap_mib\x18\n \x01(\x01\x12\x1e\n\x16peak_unpadded_heap_mib\x18\x0b \x01(\x01\x12\x1f\n\x17peak_heap_size_position\x18\x0c \x01(\x05\x12(\n entry_computation_parameters_mib\x18\r \x01(\x01\x12\x18\n\x10non_reusable_mib\x18\x0e \x01(\x01\x12\x1a\n\x12maybe_live_out_mib\x18\x0f \x01(\x01\x12\x43\n\x14indefinite_lifetimes\x18\x10 \x03(\x0b\x32%.tensorflow.profiler.BufferAllocation\x1aZ\n\x17LogicalBufferSpansEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12.\n\x05value\x18\x02 \x01(\x0b\x32\x1f.tensorflow.profiler.BufferSpan:\x02\x38\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.core.profiler.protobuf.memory_viewer_preprocess_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PREPROCESSRESULT_LOGICALBUFFERSPANSENTRY._options = None
  _PREPROCESSRESULT_LOGICALBUFFERSPANSENTRY._serialized_options = b'8\001'
  _HEAPOBJECT._serialized_start=90
  _HEAPOBJECT._serialized_end=356
  _BUFFERSPAN._serialized_start=358
  _BUFFERSPAN._serialized_end=400
  _LOGICALBUFFER._serialized_start=402
  _LOGICALBUFFER._serialized_end=501
  _BUFFERALLOCATION._serialized_start=504
  _BUFFERALLOCATION._serialized_end=655
  _PREPROCESSRESULT._serialized_start=658
  _PREPROCESSRESULT._serialized_end=1384
  _PREPROCESSRESULT_LOGICALBUFFERSPANSENTRY._serialized_start=1294
  _PREPROCESSRESULT_LOGICALBUFFERSPANSENTRY._serialized_end=1384
# @@protoc_insertion_point(module_scope)
