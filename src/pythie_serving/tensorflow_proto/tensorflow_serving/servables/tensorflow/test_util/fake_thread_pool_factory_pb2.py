# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_serving/servables/tensorflow/test_util/fake_thread_pool_factory.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_serving/servables/tensorflow/test_util/fake_thread_pool_factory.proto',
  package='tensorflow.serving.test_util',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nPtensorflow_serving/servables/tensorflow/test_util/fake_thread_pool_factory.proto\x12\x1ctensorflow.serving.test_util\"\x1d\n\x1b\x46\x61keThreadPoolFactoryConfigb\x06proto3'
)




_FAKETHREADPOOLFACTORYCONFIG = _descriptor.Descriptor(
  name='FakeThreadPoolFactoryConfig',
  full_name='tensorflow.serving.test_util.FakeThreadPoolFactoryConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
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
  serialized_start=114,
  serialized_end=143,
)

DESCRIPTOR.message_types_by_name['FakeThreadPoolFactoryConfig'] = _FAKETHREADPOOLFACTORYCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FakeThreadPoolFactoryConfig = _reflection.GeneratedProtocolMessageType('FakeThreadPoolFactoryConfig', (_message.Message,), {
  'DESCRIPTOR' : _FAKETHREADPOOLFACTORYCONFIG,
  '__module__' : 'tensorflow_serving.servables.tensorflow.test_util.fake_thread_pool_factory_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.test_util.FakeThreadPoolFactoryConfig)
  })
_sym_db.RegisterMessage(FakeThreadPoolFactoryConfig)


# @@protoc_insertion_point(module_scope)
