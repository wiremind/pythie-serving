# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/profiler/profiler_analysis.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.profiler import profiler_service_pb2 as tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n0tensorflow/core/profiler/profiler_analysis.proto\x12\ntensorflow\x1a/tensorflow/core/profiler/profiler_service.proto\"\x83\x01\n\x18NewProfileSessionRequest\x12+\n\x07request\x18\x01 \x01(\x0b\x32\x1a.tensorflow.ProfileRequest\x12\x17\n\x0frepository_root\x18\x02 \x01(\t\x12\r\n\x05hosts\x18\x03 \x03(\t\x12\x12\n\nsession_id\x18\x04 \x01(\t\"G\n\x19NewProfileSessionResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x13\n\x0b\x65mpty_trace\x18\x02 \x01(\x08\"=\n\"EnumProfileSessionsAndToolsRequest\x12\x17\n\x0frepository_root\x18\x01 \x01(\t\"A\n\x12ProfileSessionInfo\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x17\n\x0f\x61vailable_tools\x18\x02 \x03(\t\"n\n#EnumProfileSessionsAndToolsResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x30\n\x08sessions\x18\x02 \x03(\x0b\x32\x1e.tensorflow.ProfileSessionInfo\"\xec\x01\n\x19ProfileSessionDataRequest\x12\x17\n\x0frepository_root\x18\x01 \x01(\t\x12\x12\n\nsession_id\x18\x02 \x01(\t\x12\x11\n\thost_name\x18\x05 \x01(\t\x12\x11\n\ttool_name\x18\x03 \x01(\t\x12I\n\nparameters\x18\x04 \x03(\x0b\x32\x35.tensorflow.ProfileSessionDataRequest.ParametersEntry\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"Z\n\x1aProfileSessionDataResponse\x12\x15\n\rerror_message\x18\x01 \x01(\t\x12\x15\n\routput_format\x18\x02 \x01(\t\x12\x0e\n\x06output\x18\x03 \x01(\x0c\x32\xc8\x02\n\x0fProfileAnalysis\x12[\n\nNewSession\x12$.tensorflow.NewProfileSessionRequest\x1a%.tensorflow.NewProfileSessionResponse\"\x00\x12q\n\x0c\x45numSessions\x12..tensorflow.EnumProfileSessionsAndToolsRequest\x1a/.tensorflow.EnumProfileSessionsAndToolsResponse\"\x00\x12\x65\n\x12GetSessionToolData\x12%.tensorflow.ProfileSessionDataRequest\x1a&.tensorflow.ProfileSessionDataResponse\"\x00\x62\x06proto3')



_NEWPROFILESESSIONREQUEST = DESCRIPTOR.message_types_by_name['NewProfileSessionRequest']
_NEWPROFILESESSIONRESPONSE = DESCRIPTOR.message_types_by_name['NewProfileSessionResponse']
_ENUMPROFILESESSIONSANDTOOLSREQUEST = DESCRIPTOR.message_types_by_name['EnumProfileSessionsAndToolsRequest']
_PROFILESESSIONINFO = DESCRIPTOR.message_types_by_name['ProfileSessionInfo']
_ENUMPROFILESESSIONSANDTOOLSRESPONSE = DESCRIPTOR.message_types_by_name['EnumProfileSessionsAndToolsResponse']
_PROFILESESSIONDATAREQUEST = DESCRIPTOR.message_types_by_name['ProfileSessionDataRequest']
_PROFILESESSIONDATAREQUEST_PARAMETERSENTRY = _PROFILESESSIONDATAREQUEST.nested_types_by_name['ParametersEntry']
_PROFILESESSIONDATARESPONSE = DESCRIPTOR.message_types_by_name['ProfileSessionDataResponse']
NewProfileSessionRequest = _reflection.GeneratedProtocolMessageType('NewProfileSessionRequest', (_message.Message,), {
  'DESCRIPTOR' : _NEWPROFILESESSIONREQUEST,
  '__module__' : 'tensorflow.core.profiler.profiler_analysis_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.NewProfileSessionRequest)
  })
_sym_db.RegisterMessage(NewProfileSessionRequest)

NewProfileSessionResponse = _reflection.GeneratedProtocolMessageType('NewProfileSessionResponse', (_message.Message,), {
  'DESCRIPTOR' : _NEWPROFILESESSIONRESPONSE,
  '__module__' : 'tensorflow.core.profiler.profiler_analysis_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.NewProfileSessionResponse)
  })
_sym_db.RegisterMessage(NewProfileSessionResponse)

EnumProfileSessionsAndToolsRequest = _reflection.GeneratedProtocolMessageType('EnumProfileSessionsAndToolsRequest', (_message.Message,), {
  'DESCRIPTOR' : _ENUMPROFILESESSIONSANDTOOLSREQUEST,
  '__module__' : 'tensorflow.core.profiler.profiler_analysis_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.EnumProfileSessionsAndToolsRequest)
  })
_sym_db.RegisterMessage(EnumProfileSessionsAndToolsRequest)

ProfileSessionInfo = _reflection.GeneratedProtocolMessageType('ProfileSessionInfo', (_message.Message,), {
  'DESCRIPTOR' : _PROFILESESSIONINFO,
  '__module__' : 'tensorflow.core.profiler.profiler_analysis_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.ProfileSessionInfo)
  })
_sym_db.RegisterMessage(ProfileSessionInfo)

EnumProfileSessionsAndToolsResponse = _reflection.GeneratedProtocolMessageType('EnumProfileSessionsAndToolsResponse', (_message.Message,), {
  'DESCRIPTOR' : _ENUMPROFILESESSIONSANDTOOLSRESPONSE,
  '__module__' : 'tensorflow.core.profiler.profiler_analysis_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.EnumProfileSessionsAndToolsResponse)
  })
_sym_db.RegisterMessage(EnumProfileSessionsAndToolsResponse)

ProfileSessionDataRequest = _reflection.GeneratedProtocolMessageType('ProfileSessionDataRequest', (_message.Message,), {

  'ParametersEntry' : _reflection.GeneratedProtocolMessageType('ParametersEntry', (_message.Message,), {
    'DESCRIPTOR' : _PROFILESESSIONDATAREQUEST_PARAMETERSENTRY,
    '__module__' : 'tensorflow.core.profiler.profiler_analysis_pb2'
    # @@protoc_insertion_point(class_scope:tensorflow.ProfileSessionDataRequest.ParametersEntry)
    })
  ,
  'DESCRIPTOR' : _PROFILESESSIONDATAREQUEST,
  '__module__' : 'tensorflow.core.profiler.profiler_analysis_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.ProfileSessionDataRequest)
  })
_sym_db.RegisterMessage(ProfileSessionDataRequest)
_sym_db.RegisterMessage(ProfileSessionDataRequest.ParametersEntry)

ProfileSessionDataResponse = _reflection.GeneratedProtocolMessageType('ProfileSessionDataResponse', (_message.Message,), {
  'DESCRIPTOR' : _PROFILESESSIONDATARESPONSE,
  '__module__' : 'tensorflow.core.profiler.profiler_analysis_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.ProfileSessionDataResponse)
  })
_sym_db.RegisterMessage(ProfileSessionDataResponse)

_PROFILEANALYSIS = DESCRIPTOR.services_by_name['ProfileAnalysis']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PROFILESESSIONDATAREQUEST_PARAMETERSENTRY._options = None
  _PROFILESESSIONDATAREQUEST_PARAMETERSENTRY._serialized_options = b'8\001'
  _NEWPROFILESESSIONREQUEST._serialized_start=114
  _NEWPROFILESESSIONREQUEST._serialized_end=245
  _NEWPROFILESESSIONRESPONSE._serialized_start=247
  _NEWPROFILESESSIONRESPONSE._serialized_end=318
  _ENUMPROFILESESSIONSANDTOOLSREQUEST._serialized_start=320
  _ENUMPROFILESESSIONSANDTOOLSREQUEST._serialized_end=381
  _PROFILESESSIONINFO._serialized_start=383
  _PROFILESESSIONINFO._serialized_end=448
  _ENUMPROFILESESSIONSANDTOOLSRESPONSE._serialized_start=450
  _ENUMPROFILESESSIONSANDTOOLSRESPONSE._serialized_end=560
  _PROFILESESSIONDATAREQUEST._serialized_start=563
  _PROFILESESSIONDATAREQUEST._serialized_end=799
  _PROFILESESSIONDATAREQUEST_PARAMETERSENTRY._serialized_start=750
  _PROFILESESSIONDATAREQUEST_PARAMETERSENTRY._serialized_end=799
  _PROFILESESSIONDATARESPONSE._serialized_start=801
  _PROFILESESSIONDATARESPONSE._serialized_end=891
  _PROFILEANALYSIS._serialized_start=894
  _PROFILEANALYSIS._serialized_end=1222
# @@protoc_insertion_point(module_scope)
