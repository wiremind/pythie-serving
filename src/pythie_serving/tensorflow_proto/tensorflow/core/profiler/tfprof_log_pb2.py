# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/profiler/tfprof_log.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.framework import attr_value_pb2 as tensorflow_dot_core_dot_framework_dot_attr__value__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.framework import step_stats_pb2 as tensorflow_dot_core_dot_framework_dot_step__stats__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)tensorflow/core/profiler/tfprof_log.proto\x12\x11tensorflow.tfprof\x1a*tensorflow/core/framework/attr_value.proto\x1a*tensorflow/core/framework/step_stats.proto\"\xdf\x01\n\x07\x43odeDef\x12\x30\n\x06traces\x18\x01 \x03(\x0b\x32 .tensorflow.tfprof.CodeDef.Trace\x1a\xa1\x01\n\x05Trace\x12\x10\n\x04\x66ile\x18\x01 \x01(\tB\x02\x18\x01\x12\x0f\n\x07\x66ile_id\x18\x06 \x01(\x03\x12\x0e\n\x06lineno\x18\x02 \x01(\x05\x12\x14\n\x08\x66unction\x18\x03 \x01(\tB\x02\x18\x01\x12\x13\n\x0b\x66unction_id\x18\x07 \x01(\x03\x12\x10\n\x04line\x18\x04 \x01(\tB\x02\x18\x01\x12\x0f\n\x07line_id\x18\x08 \x01(\x03\x12\x17\n\x0f\x66unc_start_line\x18\x05 \x01(\x05\"j\n\nOpLogEntry\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tfloat_ops\x18\x02 \x01(\x03\x12\r\n\x05types\x18\x03 \x03(\t\x12,\n\x08\x63ode_def\x18\x04 \x01(\x0b\x32\x1a.tensorflow.tfprof.CodeDef\"\xb8\x01\n\nOpLogProto\x12\x32\n\x0blog_entries\x18\x01 \x03(\x0b\x32\x1d.tensorflow.tfprof.OpLogEntry\x12\x43\n\x0cid_to_string\x18\x02 \x03(\x0b\x32-.tensorflow.tfprof.OpLogProto.IdToStringEntry\x1a\x31\n\x0fIdToStringEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xd4\x02\n\x0cProfileProto\x12\x39\n\x05nodes\x18\x01 \x03(\x0b\x32*.tensorflow.tfprof.ProfileProto.NodesEntry\x12\x11\n\thas_trace\x18\x02 \x01(\x08\x12\x1f\n\x17miss_accelerator_stream\x18\x05 \x01(\x08\x12\r\n\x05steps\x18\x03 \x03(\x03\x12\x45\n\x0cid_to_string\x18\x04 \x03(\x0b\x32/.tensorflow.tfprof.ProfileProto.IdToStringEntry\x1aL\n\nNodesEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12-\n\x05value\x18\x02 \x01(\x0b\x32\x1e.tensorflow.tfprof.ProfileNode:\x02\x38\x01\x1a\x31\n\x0fIdToStringEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xd3\x08\n\x0bProfileNode\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\n\n\x02op\x18\t \x01(\t\x12\n\n\x02id\x18\r \x01(\x03\x12:\n\x06inputs\x18\x02 \x03(\x0b\x32*.tensorflow.tfprof.ProfileNode.InputsEntry\x12\x45\n\x0cinput_shapes\x18\x10 \x03(\x0b\x32/.tensorflow.tfprof.ProfileNode.InputShapesEntry\x12<\n\x07outputs\x18\x03 \x03(\x0b\x32+.tensorflow.tfprof.ProfileNode.OutputsEntry\x12G\n\routput_shapes\x18\x0f \x03(\x0b\x32\x30.tensorflow.tfprof.ProfileNode.OutputShapesEntry\x12L\n\x10src_output_index\x18\x0e \x03(\x0b\x32\x32.tensorflow.tfprof.ProfileNode.SrcOutputIndexEntry\x12\r\n\x05shape\x18\x04 \x03(\x03\x12\x10\n\x08op_types\x18\x05 \x03(\t\x12\x18\n\x10\x63\x61nonical_device\x18\x06 \x01(\t\x12\x13\n\x0bhost_device\x18\x07 \x01(\t\x12\x11\n\tfloat_ops\x18\x08 \x01(\x03\x12)\n\x05trace\x18\n \x01(\x0b\x32\x1a.tensorflow.tfprof.CodeDef\x12\x38\n\x05\x61ttrs\x18\x0b \x03(\x0b\x32).tensorflow.tfprof.ProfileNode.AttrsEntry\x12\x38\n\x05\x65xecs\x18\x0c \x03(\x0b\x32).tensorflow.tfprof.ProfileNode.ExecsEntry\x1a-\n\x0bInputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1aL\n\x10InputShapesEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.tensorflow.tfprof.Tuple:\x02\x38\x01\x1a.\n\x0cOutputsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1aM\n\x11OutputShapesEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\'\n\x05value\x18\x02 \x01(\x0b\x32\x18.tensorflow.tfprof.Tuple:\x02\x38\x01\x1a\x35\n\x13SrcOutputIndexEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x43\n\nAttrsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b\x32\x15.tensorflow.AttrValue:\x02\x38\x01\x1aL\n\nExecsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12-\n\x05value\x18\x02 \x01(\x0b\x32\x1e.tensorflow.tfprof.ExecProfile:\x02\x38\x01\"\x84\x04\n\x0b\x45xecProfile\x12\x11\n\trun_count\x18\x01 \x01(\x03\x12\x18\n\x10\x61ll_start_micros\x18\x02 \x01(\x03\x12\x19\n\x11latest_end_micros\x18\x03 \x01(\x03\x12O\n\x11\x61\x63\x63\x65lerator_execs\x18\x04 \x03(\x0b\x32\x34.tensorflow.tfprof.ExecProfile.AcceleratorExecsEntry\x12?\n\tcpu_execs\x18\x05 \x03(\x0b\x32,.tensorflow.tfprof.ExecProfile.CpuExecsEntry\x12\x33\n\x0cmemory_execs\x18\x07 \x03(\x0b\x32\x1d.tensorflow.tfprof.ExecMemory\x12\x31\n\x0b\x61llocations\x18\x0b \x03(\x0b\x32\x1c.tensorflow.AllocationRecord\x12\x0f\n\x07\x64\x65vices\x18\x06 \x03(\t\x1aT\n\x15\x41\x63\x63\x65leratorExecsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12*\n\x05value\x18\x02 \x01(\x0b\x32\x1b.tensorflow.tfprof.ExecTime:\x02\x38\x01\x1aL\n\rCpuExecsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12*\n\x05value\x18\x02 \x01(\x0b\x32\x1b.tensorflow.tfprof.ExecTime:\x02\x38\x01\"3\n\x08\x45xecTime\x12\'\n\x05times\x18\x01 \x03(\x0b\x32\x18.tensorflow.tfprof.Tuple\"\xb4\x03\n\nExecMemory\x12\x15\n\rmemory_micros\x18\x01 \x01(\x03\x12\x17\n\x0fhost_temp_bytes\x18\x02 \x01(\x03\x12\x1d\n\x15host_persistent_bytes\x18\x03 \x01(\x03\x12\x1e\n\x16\x61\x63\x63\x65lerator_temp_bytes\x18\x04 \x01(\x03\x12$\n\x1c\x61\x63\x63\x65lerator_persistent_bytes\x18\x05 \x01(\x03\x12\x17\n\x0frequested_bytes\x18\x06 \x01(\x03\x12\x12\n\npeak_bytes\x18\x07 \x01(\x03\x12\x16\n\x0eresidual_bytes\x18\x08 \x01(\x03\x12\x14\n\x0coutput_bytes\x18\t \x01(\x03\x12\x1e\n\x16\x61llocator_bytes_in_use\x18\n \x01(\x03\x12\x46\n\routput_memory\x18\x0b \x03(\x0b\x32/.tensorflow.tfprof.ExecMemory.OutputMemoryEntry\x1aN\n\x11OutputMemoryEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12(\n\x05value\x18\x02 \x01(\x0b\x32\x19.tensorflow.tfprof.Memory:\x02\x38\x01\"\x1d\n\x05Tuple\x12\x14\n\x0cint64_values\x18\x01 \x03(\x03\"$\n\x06Memory\x12\r\n\x05\x62ytes\x18\x01 \x01(\x03\x12\x0b\n\x03ptr\x18\x02 \x01(\x04\x42RZPgithub.com/tensorflow/tensorflow/tensorflow/go/core/profiler/tfprof_log_go_protob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.core.profiler.tfprof_log_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZPgithub.com/tensorflow/tensorflow/tensorflow/go/core/profiler/tfprof_log_go_proto'
  _CODEDEF_TRACE.fields_by_name['file']._options = None
  _CODEDEF_TRACE.fields_by_name['file']._serialized_options = b'\030\001'
  _CODEDEF_TRACE.fields_by_name['function']._options = None
  _CODEDEF_TRACE.fields_by_name['function']._serialized_options = b'\030\001'
  _CODEDEF_TRACE.fields_by_name['line']._options = None
  _CODEDEF_TRACE.fields_by_name['line']._serialized_options = b'\030\001'
  _OPLOGPROTO_IDTOSTRINGENTRY._options = None
  _OPLOGPROTO_IDTOSTRINGENTRY._serialized_options = b'8\001'
  _PROFILEPROTO_NODESENTRY._options = None
  _PROFILEPROTO_NODESENTRY._serialized_options = b'8\001'
  _PROFILEPROTO_IDTOSTRINGENTRY._options = None
  _PROFILEPROTO_IDTOSTRINGENTRY._serialized_options = b'8\001'
  _PROFILENODE_INPUTSENTRY._options = None
  _PROFILENODE_INPUTSENTRY._serialized_options = b'8\001'
  _PROFILENODE_INPUTSHAPESENTRY._options = None
  _PROFILENODE_INPUTSHAPESENTRY._serialized_options = b'8\001'
  _PROFILENODE_OUTPUTSENTRY._options = None
  _PROFILENODE_OUTPUTSENTRY._serialized_options = b'8\001'
  _PROFILENODE_OUTPUTSHAPESENTRY._options = None
  _PROFILENODE_OUTPUTSHAPESENTRY._serialized_options = b'8\001'
  _PROFILENODE_SRCOUTPUTINDEXENTRY._options = None
  _PROFILENODE_SRCOUTPUTINDEXENTRY._serialized_options = b'8\001'
  _PROFILENODE_ATTRSENTRY._options = None
  _PROFILENODE_ATTRSENTRY._serialized_options = b'8\001'
  _PROFILENODE_EXECSENTRY._options = None
  _PROFILENODE_EXECSENTRY._serialized_options = b'8\001'
  _EXECPROFILE_ACCELERATOREXECSENTRY._options = None
  _EXECPROFILE_ACCELERATOREXECSENTRY._serialized_options = b'8\001'
  _EXECPROFILE_CPUEXECSENTRY._options = None
  _EXECPROFILE_CPUEXECSENTRY._serialized_options = b'8\001'
  _EXECMEMORY_OUTPUTMEMORYENTRY._options = None
  _EXECMEMORY_OUTPUTMEMORYENTRY._serialized_options = b'8\001'
  _CODEDEF._serialized_start=153
  _CODEDEF._serialized_end=376
  _CODEDEF_TRACE._serialized_start=215
  _CODEDEF_TRACE._serialized_end=376
  _OPLOGENTRY._serialized_start=378
  _OPLOGENTRY._serialized_end=484
  _OPLOGPROTO._serialized_start=487
  _OPLOGPROTO._serialized_end=671
  _OPLOGPROTO_IDTOSTRINGENTRY._serialized_start=622
  _OPLOGPROTO_IDTOSTRINGENTRY._serialized_end=671
  _PROFILEPROTO._serialized_start=674
  _PROFILEPROTO._serialized_end=1014
  _PROFILEPROTO_NODESENTRY._serialized_start=887
  _PROFILEPROTO_NODESENTRY._serialized_end=963
  _PROFILEPROTO_IDTOSTRINGENTRY._serialized_start=622
  _PROFILEPROTO_IDTOSTRINGENTRY._serialized_end=671
  _PROFILENODE._serialized_start=1017
  _PROFILENODE._serialized_end=2124
  _PROFILENODE_INPUTSENTRY._serialized_start=1672
  _PROFILENODE_INPUTSENTRY._serialized_end=1717
  _PROFILENODE_INPUTSHAPESENTRY._serialized_start=1719
  _PROFILENODE_INPUTSHAPESENTRY._serialized_end=1795
  _PROFILENODE_OUTPUTSENTRY._serialized_start=1797
  _PROFILENODE_OUTPUTSENTRY._serialized_end=1843
  _PROFILENODE_OUTPUTSHAPESENTRY._serialized_start=1845
  _PROFILENODE_OUTPUTSHAPESENTRY._serialized_end=1922
  _PROFILENODE_SRCOUTPUTINDEXENTRY._serialized_start=1924
  _PROFILENODE_SRCOUTPUTINDEXENTRY._serialized_end=1977
  _PROFILENODE_ATTRSENTRY._serialized_start=1979
  _PROFILENODE_ATTRSENTRY._serialized_end=2046
  _PROFILENODE_EXECSENTRY._serialized_start=2048
  _PROFILENODE_EXECSENTRY._serialized_end=2124
  _EXECPROFILE._serialized_start=2127
  _EXECPROFILE._serialized_end=2643
  _EXECPROFILE_ACCELERATOREXECSENTRY._serialized_start=2481
  _EXECPROFILE_ACCELERATOREXECSENTRY._serialized_end=2565
  _EXECPROFILE_CPUEXECSENTRY._serialized_start=2567
  _EXECPROFILE_CPUEXECSENTRY._serialized_end=2643
  _EXECTIME._serialized_start=2645
  _EXECTIME._serialized_end=2696
  _EXECMEMORY._serialized_start=2699
  _EXECMEMORY._serialized_end=3135
  _EXECMEMORY_OUTPUTMEMORYENTRY._serialized_start=3057
  _EXECMEMORY_OUTPUTMEMORYENTRY._serialized_end=3135
  _TUPLE._serialized_start=3137
  _TUPLE._serialized_end=3166
  _MEMORY._serialized_start=3168
  _MEMORY._serialized_end=3204
# @@protoc_insertion_point(module_scope)
