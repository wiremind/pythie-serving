# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/compiler/xla/pjrt/compile_options.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.compiler.xla import xla_pb2 as tensorflow_dot_compiler_dot_xla_dot_xla__pb2
from pythie_serving.tensorflow_proto.tensorflow.compiler.xla import xla_data_pb2 as tensorflow_dot_compiler_dot_xla_dot_xla__data__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n2tensorflow/compiler/xla/pjrt/compile_options.proto\x12\x03xla\x1a!tensorflow/compiler/xla/xla.proto\x1a&tensorflow/compiler/xla/xla_data.proto\"\xb7\x03\n\x1b\x45xecutableBuildOptionsProto\x12\x16\n\x0e\x64\x65vice_ordinal\x18\x01 \x01(\x03\x12&\n\rresult_layout\x18\x02 \x01(\x0b\x32\x0f.xla.ShapeProto\x12(\n\rdebug_options\x18\x03 \x01(\x0b\x32\x11.xla.DebugOptions\x12\x14\n\x0cnum_replicas\x18\x04 \x01(\x03\x12\x16\n\x0enum_partitions\x18\x05 \x01(\x03\x12\x1d\n\x15use_spmd_partitioning\x18\x06 \x01(\x08\x12\"\n\x1ause_auto_spmd_partitioning\x18\x07 \x01(\x08\x12\x17\n\x0f\x64\x65\x64uplicate_hlo\x18\x08 \x01(\x08\x12\x35\n\x11\x64\x65vice_assignment\x18\t \x01(\x0b\x32\x1a.xla.DeviceAssignmentProto\x12 \n\x18\x61lias_passthrough_params\x18\n \x01(\x08\x12\x18\n\x10run_backend_only\x18\x0b \x01(\x08\x12\x31\n)allow_spmd_sharding_propagation_to_output\x18\x0c \x01(\x08\"\x90\x02\n\x13\x43ompileOptionsProto\x12)\n\x10\x61rgument_layouts\x18\x01 \x03(\x0b\x32\x0f.xla.ShapeProto\x12%\n\x1dparameter_is_tupled_arguments\x18\x02 \x01(\x08\x12\x42\n\x18\x65xecutable_build_options\x18\x03 \x01(\x0b\x32 .xla.ExecutableBuildOptionsProto\x12#\n\x1b\x63ompile_portable_executable\x18\x04 \x01(\x08\x12\x17\n\x0fprofile_version\x18\x05 \x01(\x03\x12%\n\x1dserialized_multi_slice_config\x18\x06 \x01(\x0c\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.compiler.xla.pjrt.compile_options_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _EXECUTABLEBUILDOPTIONSPROTO._serialized_start=135
  _EXECUTABLEBUILDOPTIONSPROTO._serialized_end=574
  _COMPILEOPTIONSPROTO._serialized_start=577
  _COMPILEOPTIONSPROTO._serialized_end=849
# @@protoc_insertion_point(module_scope)
