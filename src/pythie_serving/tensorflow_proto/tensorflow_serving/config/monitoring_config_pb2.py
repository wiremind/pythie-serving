# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_serving/config/monitoring_config.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n1tensorflow_serving/config/monitoring_config.proto\x12\x12tensorflow.serving\"0\n\x10PrometheusConfig\x12\x0e\n\x06\x65nable\x18\x01 \x01(\x08\x12\x0c\n\x04path\x18\x02 \x01(\t\"S\n\x10MonitoringConfig\x12?\n\x11prometheus_config\x18\x01 \x01(\x0b\x32$.tensorflow.serving.PrometheusConfigB\x03\xf8\x01\x01\x62\x06proto3')



_PROMETHEUSCONFIG = DESCRIPTOR.message_types_by_name['PrometheusConfig']
_MONITORINGCONFIG = DESCRIPTOR.message_types_by_name['MonitoringConfig']
PrometheusConfig = _reflection.GeneratedProtocolMessageType('PrometheusConfig', (_message.Message,), {
  'DESCRIPTOR' : _PROMETHEUSCONFIG,
  '__module__' : 'tensorflow_serving.config.monitoring_config_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.PrometheusConfig)
  })
_sym_db.RegisterMessage(PrometheusConfig)

MonitoringConfig = _reflection.GeneratedProtocolMessageType('MonitoringConfig', (_message.Message,), {
  'DESCRIPTOR' : _MONITORINGCONFIG,
  '__module__' : 'tensorflow_serving.config.monitoring_config_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.MonitoringConfig)
  })
_sym_db.RegisterMessage(MonitoringConfig)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _PROMETHEUSCONFIG._serialized_start=73
  _PROMETHEUSCONFIG._serialized_end=121
  _MONITORINGCONFIG._serialized_start=123
  _MONITORINGCONFIG._serialized_end=206
# @@protoc_insertion_point(module_scope)
