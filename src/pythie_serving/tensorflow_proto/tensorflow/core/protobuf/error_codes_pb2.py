# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/protobuf/error_codes.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*tensorflow/core/protobuf/error_codes.proto\x12\x10tensorflow.error*\x84\x03\n\x04\x43ode\x12\x06\n\x02OK\x10\x00\x12\r\n\tCANCELLED\x10\x01\x12\x0b\n\x07UNKNOWN\x10\x02\x12\x14\n\x10INVALID_ARGUMENT\x10\x03\x12\x15\n\x11\x44\x45\x41\x44LINE_EXCEEDED\x10\x04\x12\r\n\tNOT_FOUND\x10\x05\x12\x12\n\x0e\x41LREADY_EXISTS\x10\x06\x12\x15\n\x11PERMISSION_DENIED\x10\x07\x12\x13\n\x0fUNAUTHENTICATED\x10\x10\x12\x16\n\x12RESOURCE_EXHAUSTED\x10\x08\x12\x17\n\x13\x46\x41ILED_PRECONDITION\x10\t\x12\x0b\n\x07\x41\x42ORTED\x10\n\x12\x10\n\x0cOUT_OF_RANGE\x10\x0b\x12\x11\n\rUNIMPLEMENTED\x10\x0c\x12\x0c\n\x08INTERNAL\x10\r\x12\x0f\n\x0bUNAVAILABLE\x10\x0e\x12\r\n\tDATA_LOSS\x10\x0f\x12K\nGDO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_\x10\x14\x42\x88\x01\n\x18org.tensorflow.frameworkB\x10\x45rrorCodesProtosP\x01ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\xf8\x01\x01\x62\x06proto3')

_CODE = DESCRIPTOR.enum_types_by_name['Code']
Code = enum_type_wrapper.EnumTypeWrapper(_CODE)
OK = 0
CANCELLED = 1
UNKNOWN = 2
INVALID_ARGUMENT = 3
DEADLINE_EXCEEDED = 4
NOT_FOUND = 5
ALREADY_EXISTS = 6
PERMISSION_DENIED = 7
UNAUTHENTICATED = 16
RESOURCE_EXHAUSTED = 8
FAILED_PRECONDITION = 9
ABORTED = 10
OUT_OF_RANGE = 11
UNIMPLEMENTED = 12
INTERNAL = 13
UNAVAILABLE = 14
DATA_LOSS = 15
DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_ = 20


if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\030org.tensorflow.frameworkB\020ErrorCodesProtosP\001ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\370\001\001'
  _CODE._serialized_start=65
  _CODE._serialized_end=453
# @@protoc_insertion_point(module_scope)
