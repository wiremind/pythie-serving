# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/compiler/xla/service/cpu/executable.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.compiler.xla.service.cpu import xla_framework_pb2 as tensorflow_dot_compiler_dot_xla_dot_service_dot_cpu_dot_xla__framework__pb2
from pythie_serving.tensorflow_proto.tensorflow.compiler.xla.service import hlo_pb2 as tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n4tensorflow/compiler/xla/service/cpu/executable.proto\x12\x07xla.cpu\x1a\x37tensorflow/compiler/xla/service/cpu/xla_framework.proto\x1a)tensorflow/compiler/xla/service/hlo.proto\"\xd7\x01\n\x1cXlaRuntimeCpuExecutableProto\x12>\n\x16xla_runtime_executable\x18\x01 \x01(\x0b\x32\x1e.xla.XlaRuntimeExecutableProto\x12@\n\x15xla_framework_mapping\x18\x02 \x01(\x0b\x32!.xla.cpu.XlaFrameworkMappingProto\x12\x35\n\x11\x62uffer_assignment\x18\x03 \x01(\x0b\x32\x1a.xla.BufferAssignmentProto')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.compiler.xla.service.cpu.executable_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _XLARUNTIMECPUEXECUTABLEPROTO._serialized_start=166
  _XLARUNTIMECPUEXECUTABLEPROTO._serialized_end=381
# @@protoc_insertion_point(module_scope)
