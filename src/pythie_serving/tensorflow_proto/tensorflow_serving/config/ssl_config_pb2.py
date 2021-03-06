# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_serving/config/ssl_config.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_serving/config/ssl_config.proto',
  package='tensorflow.serving',
  syntax='proto3',
  serialized_options=b'\370\001\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n*tensorflow_serving/config/ssl_config.proto\x12\x12tensorflow.serving\"^\n\tSSLConfig\x12\x12\n\nserver_key\x18\x01 \x01(\t\x12\x13\n\x0bserver_cert\x18\x02 \x01(\t\x12\x11\n\tcustom_ca\x18\x03 \x01(\t\x12\x15\n\rclient_verify\x18\x04 \x01(\x08\x42\x03\xf8\x01\x01\x62\x06proto3'
)




_SSLCONFIG = _descriptor.Descriptor(
  name='SSLConfig',
  full_name='tensorflow.serving.SSLConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='server_key', full_name='tensorflow.serving.SSLConfig.server_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='server_cert', full_name='tensorflow.serving.SSLConfig.server_cert', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='custom_ca', full_name='tensorflow.serving.SSLConfig.custom_ca', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='client_verify', full_name='tensorflow.serving.SSLConfig.client_verify', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=66,
  serialized_end=160,
)

DESCRIPTOR.message_types_by_name['SSLConfig'] = _SSLCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SSLConfig = _reflection.GeneratedProtocolMessageType('SSLConfig', (_message.Message,), {
  'DESCRIPTOR' : _SSLCONFIG,
  '__module__' : 'tensorflow_serving.config.ssl_config_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.SSLConfig)
  })
_sym_db.RegisterMessage(SSLConfig)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
