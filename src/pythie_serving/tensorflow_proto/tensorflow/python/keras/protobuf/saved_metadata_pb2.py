# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/python/keras/protobuf/saved_metadata.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.python.keras.protobuf import versions_pb2 as tensorflow_dot_python_dot_keras_dot_protobuf_dot_versions__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n5tensorflow/python/keras/protobuf/saved_metadata.proto\x12,third_party.tensorflow.python.keras.protobuf\x1a/tensorflow/python/keras/protobuf/versions.proto\"Y\n\rSavedMetadata\x12H\n\x05nodes\x18\x01 \x03(\x0b\x32\x39.third_party.tensorflow.python.keras.protobuf.SavedObject\"\xa8\x01\n\x0bSavedObject\x12\x0f\n\x07node_id\x18\x02 \x01(\x05\x12\x11\n\tnode_path\x18\x03 \x01(\t\x12\x12\n\nidentifier\x18\x04 \x01(\t\x12\x10\n\x08metadata\x18\x05 \x01(\t\x12I\n\x07version\x18\x06 \x01(\x0b\x32\x38.third_party.tensorflow.python.keras.protobuf.VersionDefJ\x04\x08\x01\x10\x02\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.python.keras.protobuf.saved_metadata_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _SAVEDMETADATA._serialized_start=152
  _SAVEDMETADATA._serialized_end=241
  _SAVEDOBJECT._serialized_start=244
  _SAVEDOBJECT._serialized_end=412
# @@protoc_insertion_point(module_scope)
