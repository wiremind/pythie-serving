# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/device_attributes.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n1tensorflow/core/framework/device_attributes.proto\x12\ntensorflow\"E\n\x10InterconnectLink\x12\x11\n\tdevice_id\x18\x01 \x01(\x05\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x10\n\x08strength\x18\x03 \x01(\x05\"8\n\nLocalLinks\x12*\n\x04link\x18\x01 \x03(\x0b\x32\x1c.tensorflow.InterconnectLink\"Z\n\x0e\x44\x65viceLocality\x12\x0e\n\x06\x62us_id\x18\x01 \x01(\x05\x12\x11\n\tnuma_node\x18\x02 \x01(\x05\x12%\n\x05links\x18\x03 \x01(\x0b\x32\x16.tensorflow.LocalLinks\"\xac\x01\n\x10\x44\x65viceAttributes\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x65vice_type\x18\x02 \x01(\t\x12\x14\n\x0cmemory_limit\x18\x04 \x01(\x03\x12,\n\x08locality\x18\x05 \x01(\x0b\x32\x1a.tensorflow.DeviceLocality\x12\x13\n\x0bincarnation\x18\x06 \x01(\x06\x12\x1c\n\x14physical_device_desc\x18\x07 \x01(\tB\x91\x01\n\x18org.tensorflow.frameworkB\x16\x44\x65viceAttributesProtosP\x01ZXgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/device_attributes_go_proto\xf8\x01\x01\x62\x06proto3')



_INTERCONNECTLINK = DESCRIPTOR.message_types_by_name['InterconnectLink']
_LOCALLINKS = DESCRIPTOR.message_types_by_name['LocalLinks']
_DEVICELOCALITY = DESCRIPTOR.message_types_by_name['DeviceLocality']
_DEVICEATTRIBUTES = DESCRIPTOR.message_types_by_name['DeviceAttributes']
InterconnectLink = _reflection.GeneratedProtocolMessageType('InterconnectLink', (_message.Message,), {
  'DESCRIPTOR' : _INTERCONNECTLINK,
  '__module__' : 'tensorflow.core.framework.device_attributes_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.InterconnectLink)
  })
_sym_db.RegisterMessage(InterconnectLink)

LocalLinks = _reflection.GeneratedProtocolMessageType('LocalLinks', (_message.Message,), {
  'DESCRIPTOR' : _LOCALLINKS,
  '__module__' : 'tensorflow.core.framework.device_attributes_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.LocalLinks)
  })
_sym_db.RegisterMessage(LocalLinks)

DeviceLocality = _reflection.GeneratedProtocolMessageType('DeviceLocality', (_message.Message,), {
  'DESCRIPTOR' : _DEVICELOCALITY,
  '__module__' : 'tensorflow.core.framework.device_attributes_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.DeviceLocality)
  })
_sym_db.RegisterMessage(DeviceLocality)

DeviceAttributes = _reflection.GeneratedProtocolMessageType('DeviceAttributes', (_message.Message,), {
  'DESCRIPTOR' : _DEVICEATTRIBUTES,
  '__module__' : 'tensorflow.core.framework.device_attributes_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.DeviceAttributes)
  })
_sym_db.RegisterMessage(DeviceAttributes)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\030org.tensorflow.frameworkB\026DeviceAttributesProtosP\001ZXgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/device_attributes_go_proto\370\001\001'
  _INTERCONNECTLINK._serialized_start=65
  _INTERCONNECTLINK._serialized_end=134
  _LOCALLINKS._serialized_start=136
  _LOCALLINKS._serialized_end=192
  _DEVICELOCALITY._serialized_start=194
  _DEVICELOCALITY._serialized_end=284
  _DEVICEATTRIBUTES._serialized_start=287
  _DEVICEATTRIBUTES._serialized_end=459
# @@protoc_insertion_point(module_scope)
