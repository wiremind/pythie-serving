# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/compiler/jit/xla_compilation_cache.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.compiler.xla.service import hlo_pb2 as tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n3tensorflow/compiler/jit/xla_compilation_cache.proto\x12\ntensorflow\x1a)tensorflow/compiler/xla/service/hlo.proto\"x\n\x15XlaSerializedCacheKey\x12\x1d\n\x15signature_fingerprint\x18\x01 \x01(\x04\x12\x1b\n\x13\x63luster_fingerprint\x18\x02 \x01(\x04\x12\x13\n\x0b\x64\x65vice_type\x18\x03 \x01(\t\x12\x0e\n\x06prefix\x18\x04 \x01(\t\"\x86\x01\n\x17XlaSerializedCacheEntry\x12.\n\x03key\x18\x01 \x01(\x0b\x32!.tensorflow.XlaSerializedCacheKey\x12\'\n\nhlo_module\x18\x02 \x01(\x0b\x32\x13.xla.HloModuleProto\x12\x12\n\nexecutable\x18\x03 \x01(\x0c\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.compiler.jit.xla_compilation_cache_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _XLASERIALIZEDCACHEKEY._serialized_start=110
  _XLASERIALIZEDCACHEKEY._serialized_end=230
  _XLASERIALIZEDCACHEENTRY._serialized_start=233
  _XLASERIALIZEDCACHEENTRY._serialized_end=367
# @@protoc_insertion_point(module_scope)