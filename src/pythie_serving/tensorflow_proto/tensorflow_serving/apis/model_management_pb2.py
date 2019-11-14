# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_serving/apis/model_management.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow_serving.config import model_server_config_pb2 as tensorflow__serving_dot_config_dot_model__server__config__pb2
from pythie_serving.tensorflow_proto.tensorflow_serving.util import status_pb2 as tensorflow__serving_dot_util_dot_status__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_serving/apis/model_management.proto',
  package='tensorflow.serving',
  syntax='proto3',
  serialized_options=_b('\370\001\001'),
  serialized_pb=_b('\n.tensorflow_serving/apis/model_management.proto\x12\x12tensorflow.serving\x1a\x33tensorflow_serving/config/model_server_config.proto\x1a$tensorflow_serving/util/status.proto\"L\n\x13ReloadConfigRequest\x12\x35\n\x06\x63onfig\x18\x01 \x01(\x0b\x32%.tensorflow.serving.ModelServerConfig\"G\n\x14ReloadConfigResponse\x12/\n\x06status\x18\x01 \x01(\x0b\x32\x1f.tensorflow.serving.StatusProtoB\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow__serving_dot_config_dot_model__server__config__pb2.DESCRIPTOR,tensorflow__serving_dot_util_dot_status__pb2.DESCRIPTOR,])




_RELOADCONFIGREQUEST = _descriptor.Descriptor(
  name='ReloadConfigRequest',
  full_name='tensorflow.serving.ReloadConfigRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='config', full_name='tensorflow.serving.ReloadConfigRequest.config', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_end=237,
)


_RELOADCONFIGRESPONSE = _descriptor.Descriptor(
  name='ReloadConfigResponse',
  full_name='tensorflow.serving.ReloadConfigResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='tensorflow.serving.ReloadConfigResponse.status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=239,
  serialized_end=310,
)

_RELOADCONFIGREQUEST.fields_by_name['config'].message_type = tensorflow__serving_dot_config_dot_model__server__config__pb2._MODELSERVERCONFIG
_RELOADCONFIGRESPONSE.fields_by_name['status'].message_type = tensorflow__serving_dot_util_dot_status__pb2._STATUSPROTO
DESCRIPTOR.message_types_by_name['ReloadConfigRequest'] = _RELOADCONFIGREQUEST
DESCRIPTOR.message_types_by_name['ReloadConfigResponse'] = _RELOADCONFIGRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ReloadConfigRequest = _reflection.GeneratedProtocolMessageType('ReloadConfigRequest', (_message.Message,), {
  'DESCRIPTOR' : _RELOADCONFIGREQUEST,
  '__module__' : 'tensorflow_serving.apis.model_management_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ReloadConfigRequest)
  })
_sym_db.RegisterMessage(ReloadConfigRequest)

ReloadConfigResponse = _reflection.GeneratedProtocolMessageType('ReloadConfigResponse', (_message.Message,), {
  'DESCRIPTOR' : _RELOADCONFIGRESPONSE,
  '__module__' : 'tensorflow_serving.apis.model_management_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ReloadConfigResponse)
  })
_sym_db.RegisterMessage(ReloadConfigResponse)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
