# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/compiler/xla/service/gpu/gpu_autotuning.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.compiler.xla.service import hlo_pb2 as tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2
from pythie_serving.tensorflow_proto.tensorflow.compiler.xla import xla_data_pb2 as tensorflow_dot_compiler_dot_xla_dot_xla__data__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.protobuf import autotuning_pb2 as tensorflow_dot_core_dot_protobuf_dot_autotuning__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n8tensorflow/compiler/xla/service/gpu/gpu_autotuning.proto\x12\x07xla.gpu\x1a)tensorflow/compiler/xla/service/hlo.proto\x1a&tensorflow/compiler/xla/xla_data.proto\x1a)tensorflow/core/protobuf/autotuning.proto\"\x9f\x01\n\x12\x43onvInstructionLog\x12-\n\x0binstruction\x18\x01 \x01(\x0b\x32\x18.xla.HloInstructionProto\x12\'\n\x0eoperand_shapes\x18\x02 \x03(\x0b\x32\x0f.xla.ShapeProto\x12\x16\n\x0eresult_address\x18\x03 \x01(\x04\x12\x19\n\x11operand_addresses\x18\x04 \x03(\x04\"5\n\x13\x44\x65nylistedAlgorithm\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x12\n\ntensor_ops\x18\x02 \x01(\x08\"\xc4\x01\n\x16\x41lgorithmDenylistEntry\x12\x0b\n\x03hlo\x18\x01 \x01(\t\x12)\n\x02\x63\x63\x18\x02 \x01(\x0b\x32\x1d.tensorflow.ComputeCapability\x12/\n\rcudnn_version\x18\x03 \x01(\x0b\x32\x18.tensorflow.CudnnVersion\x12\x14\n\x0c\x62las_version\x18\x05 \x01(\t\x12+\n\x05\x61lgos\x18\x04 \x03(\x0b\x32\x1c.xla.gpu.DenylistedAlgorithm\"E\n\x11\x41lgorithmDenylist\x12\x30\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x1f.xla.gpu.AlgorithmDenylistEntryb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.compiler.xla.service.gpu.gpu_autotuning_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CONVINSTRUCTIONLOG._serialized_start=196
  _CONVINSTRUCTIONLOG._serialized_end=355
  _DENYLISTEDALGORITHM._serialized_start=357
  _DENYLISTEDALGORITHM._serialized_end=410
  _ALGORITHMDENYLISTENTRY._serialized_start=413
  _ALGORITHMDENYLISTENTRY._serialized_end=609
  _ALGORITHMDENYLIST._serialized_start=611
  _ALGORITHMDENYLIST._serialized_end=680
# @@protoc_insertion_point(module_scope)
