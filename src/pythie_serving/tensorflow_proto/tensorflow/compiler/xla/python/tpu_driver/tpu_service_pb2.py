# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/compiler/xla/python/tpu_driver/tpu_service.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.compiler.xla.python.tpu_driver import tpu_driver_pb2 as tensorflow_dot_compiler_dot_xla_dot_python_dot_tpu__driver_dot_tpu__driver__pb2
from pythie_serving.tensorflow_proto.tensorflow.compiler.xla.service import hlo_pb2 as tensorflow_dot_compiler_dot_xla_dot_service_dot_hlo__pb2
from pythie_serving.tensorflow_proto.tensorflow.compiler.xla import xla_pb2 as tensorflow_dot_compiler_dot_xla_dot_xla__pb2
from pythie_serving.tensorflow_proto.tensorflow.compiler.xla import xla_data_pb2 as tensorflow_dot_compiler_dot_xla_dot_xla__data__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n;tensorflow/compiler/xla/python/tpu_driver/tpu_service.proto\x12\ntpu_driver\x1a:tensorflow/compiler/xla/python/tpu_driver/tpu_driver.proto\x1a)tensorflow/compiler/xla/service/hlo.proto\x1a!tensorflow/compiler/xla/xla.proto\x1a&tensorflow/compiler/xla/xla_data.proto\".\n\rStatusMessage\x12\x0c\n\x04\x63ode\x18\x01 \x02(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\"\x8b\x01\n\x0f\x41llocateRequest\x12\x0f\n\x07\x63ore_id\x18\x01 \x02(\x05\x12(\n\x06region\x18\x02 \x02(\x0e\x32\x18.tpu_driver.MemoryRegion\x12\x13\n\tnum_bytes\x18\x03 \x01(\x03H\x00\x12 \n\x05shape\x18\x04 \x01(\x0b\x32\x0f.xla.ShapeProtoH\x00\x42\x06\n\x04size\"c\n\x14\x41llocateTupleRequest\x12\x0f\n\x07\x63ore_id\x18\x01 \x02(\x05\x12(\n\x06region\x18\x02 \x02(\x0e\x32\x18.tpu_driver.MemoryRegion\x12\x10\n\x08\x63hildren\x18\x03 \x03(\x03\"#\n\x11\x44\x65\x61llocateRequest\x12\x0e\n\x06handle\x18\x01 \x02(\x03\">\n\x17TransferToDeviceRequest\x12\x15\n\rtarget_handle\x18\x01 \x02(\x03\x12\x0c\n\x04\x64\x61ta\x18\x02 \x02(\x0c\"2\n\x19TransferFromDeviceRequest\x12\x15\n\rsource_handle\x18\x01 \x02(\x03\"*\n\x1aTransferFromDeviceResponse\x12\x0c\n\x04\x64\x61ta\x18\x02 \x02(\x0c\"Q\n!TransferFromDeviceToDeviceRequest\x12\x15\n\rsource_handle\x18\x01 \x02(\x03\x12\x15\n\rtarget_handle\x18\x02 \x02(\x03\"t\n\x0e\x43ompileRequest\x12\"\n\x0bhlo_program\x18\x01 \x02(\x0b\x32\r.xla.HloProto\x12\x14\n\x0cnum_replicas\x18\x02 \x01(\x03\x12(\n\rdebug_options\x18\x03 \x01(\x0b\x32\x11.xla.DebugOptions\"H\n\x17\x43ompiledProgramMetadata\x12-\n\rprogram_shape\x18\x01 \x02(\x0b\x32\x16.xla.ProgramShapeProto\"H\n\x0f\x43ompileResponse\x12\x35\n\x08metadata\x18\x01 \x02(\x0b\x32#.tpu_driver.CompiledProgramMetadata\"F\n\x12LoadProgramRequest\x12\x0f\n\x07\x63ore_id\x18\x01 \x02(\x05\x12\x1f\n\x17\x63ompiled_program_handle\x18\x02 \x02(\x03\"5\n\x14UnloadProgramRequest\x12\x1d\n\x15loaded_program_handle\x18\x01 \x02(\x03\"\x93\x01\n\x0e\x45xecuteRequest\x12\x1d\n\x15loaded_program_handle\x18\x01 \x02(\x03\x12\x14\n\x0cinput_handle\x18\x02 \x03(\x03\x12\x15\n\routput_handle\x18\x03 \x03(\x03\x12\x35\n\x11\x64\x65vice_assignment\x18\x04 \x01(\x0b\x32\x1a.xla.DeviceAssignmentProto\"\xb4\x05\n\rStreamRequest\x12.\n\x05\x65ntry\x18\x1e \x03(\x0b\x32\x1f.tpu_driver.StreamRequest.Entry\x1a\xf2\x04\n\x05\x45ntry\x12,\n\x05\x61lloc\x18\x01 \x01(\x0b\x32\x1b.tpu_driver.AllocateRequestH\x00\x12\x37\n\x0b\x61lloc_tuple\x18\x02 \x01(\x0b\x32 .tpu_driver.AllocateTupleRequestH\x00\x12\x30\n\x07\x64\x65\x61lloc\x18\x03 \x01(\x0b\x32\x1d.tpu_driver.DeallocateRequestH\x00\x12:\n\x0btransfer_to\x18\x04 \x01(\x0b\x32#.tpu_driver.TransferToDeviceRequestH\x00\x12>\n\rtransfer_from\x18\x05 \x01(\x0b\x32%.tpu_driver.TransferFromDeviceRequestH\x00\x12I\n\x10transfer_from_to\x18\n \x01(\x0b\x32-.tpu_driver.TransferFromDeviceToDeviceRequestH\x00\x12-\n\x07\x63ompile\x18\x06 \x01(\x0b\x32\x1a.tpu_driver.CompileRequestH\x00\x12.\n\x04load\x18\x07 \x01(\x0b\x32\x1e.tpu_driver.LoadProgramRequestH\x00\x12\x32\n\x06unload\x18\x08 \x01(\x0b\x32 .tpu_driver.UnloadProgramRequestH\x00\x12-\n\x07\x65xecute\x18\t \x01(\x0b\x32\x1a.tpu_driver.ExecuteRequestH\x00\x12\x13\n\x0bwait_for_id\x18\x14 \x03(\x03\x12\x14\n\x0coperation_id\x18\x15 \x02(\x03\x12\x11\n\tthread_id\x18\x16 \x01(\x03\x42\t\n\x07request\"\x89\x02\n\x0eStreamResponse\x12/\n\x05\x65ntry\x18\x14 \x03(\x0b\x32 .tpu_driver.StreamResponse.Entry\x1a\xc5\x01\n\x05\x45ntry\x12?\n\rtransfer_from\x18\x03 \x01(\x0b\x32&.tpu_driver.TransferFromDeviceResponseH\x00\x12.\n\x07\x63ompile\x18\x04 \x01(\x0b\x32\x1b.tpu_driver.CompileResponseH\x00\x12)\n\x06status\x18\n \x02(\x0b\x32\x19.tpu_driver.StatusMessage\x12\x14\n\x0coperation_id\x18\x0b \x02(\x03\x42\n\n\x08response\"(\n\x0bOpenRequest\x12\x19\n\x0e\x63lient_version\x18\x01 \x01(\x05:\x01\x30\"F\n\x0cOpenResponse\x12\x11\n\tclient_id\x18\x01 \x02(\r\x12#\n\x15max_idle_time_seconds\x18\x02 \x01(\x05:\x04\x33\x36\x30\x30\"!\n\x0c\x43loseRequest\x12\x11\n\tclient_id\x18\x01 \x02(\x07\"\x0f\n\rCloseResponse\"\x0e\n\x0cResetRequest\"\x0f\n\rResetResponse\"\x18\n\x16QuerySystemInfoRequest\"F\n\x17QuerySystemInfoResponse\x12+\n\x0bsystem_info\x18\x01 \x02(\x0b\x32\x16.tpu_driver.SystemInfo2\xef\x02\n\x0e\x43loudTpuDriver\x12\x39\n\x04Open\x12\x17.tpu_driver.OpenRequest\x1a\x18.tpu_driver.OpenResponse\x12<\n\x05\x43lose\x12\x18.tpu_driver.CloseRequest\x1a\x19.tpu_driver.CloseResponse\x12<\n\x05Reset\x12\x18.tpu_driver.ResetRequest\x1a\x19.tpu_driver.ResetResponse\x12Z\n\x0fQuerySystemInfo\x12\".tpu_driver.QuerySystemInfoRequest\x1a#.tpu_driver.QuerySystemInfoResponse\x12J\n\rStreamExecute\x12\x19.tpu_driver.StreamRequest\x1a\x1a.tpu_driver.StreamResponse(\x01\x30\x01\x42\x02H\x01')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.compiler.xla.python.tpu_driver.tpu_service_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'H\001'
  _STATUSMESSAGE._serialized_start=253
  _STATUSMESSAGE._serialized_end=299
  _ALLOCATEREQUEST._serialized_start=302
  _ALLOCATEREQUEST._serialized_end=441
  _ALLOCATETUPLEREQUEST._serialized_start=443
  _ALLOCATETUPLEREQUEST._serialized_end=542
  _DEALLOCATEREQUEST._serialized_start=544
  _DEALLOCATEREQUEST._serialized_end=579
  _TRANSFERTODEVICEREQUEST._serialized_start=581
  _TRANSFERTODEVICEREQUEST._serialized_end=643
  _TRANSFERFROMDEVICEREQUEST._serialized_start=645
  _TRANSFERFROMDEVICEREQUEST._serialized_end=695
  _TRANSFERFROMDEVICERESPONSE._serialized_start=697
  _TRANSFERFROMDEVICERESPONSE._serialized_end=739
  _TRANSFERFROMDEVICETODEVICEREQUEST._serialized_start=741
  _TRANSFERFROMDEVICETODEVICEREQUEST._serialized_end=822
  _COMPILEREQUEST._serialized_start=824
  _COMPILEREQUEST._serialized_end=940
  _COMPILEDPROGRAMMETADATA._serialized_start=942
  _COMPILEDPROGRAMMETADATA._serialized_end=1014
  _COMPILERESPONSE._serialized_start=1016
  _COMPILERESPONSE._serialized_end=1088
  _LOADPROGRAMREQUEST._serialized_start=1090
  _LOADPROGRAMREQUEST._serialized_end=1160
  _UNLOADPROGRAMREQUEST._serialized_start=1162
  _UNLOADPROGRAMREQUEST._serialized_end=1215
  _EXECUTEREQUEST._serialized_start=1218
  _EXECUTEREQUEST._serialized_end=1365
  _STREAMREQUEST._serialized_start=1368
  _STREAMREQUEST._serialized_end=2060
  _STREAMREQUEST_ENTRY._serialized_start=1434
  _STREAMREQUEST_ENTRY._serialized_end=2060
  _STREAMRESPONSE._serialized_start=2063
  _STREAMRESPONSE._serialized_end=2328
  _STREAMRESPONSE_ENTRY._serialized_start=2131
  _STREAMRESPONSE_ENTRY._serialized_end=2328
  _OPENREQUEST._serialized_start=2330
  _OPENREQUEST._serialized_end=2370
  _OPENRESPONSE._serialized_start=2372
  _OPENRESPONSE._serialized_end=2442
  _CLOSEREQUEST._serialized_start=2444
  _CLOSEREQUEST._serialized_end=2477
  _CLOSERESPONSE._serialized_start=2479
  _CLOSERESPONSE._serialized_end=2494
  _RESETREQUEST._serialized_start=2496
  _RESETREQUEST._serialized_end=2510
  _RESETRESPONSE._serialized_start=2512
  _RESETRESPONSE._serialized_end=2527
  _QUERYSYSTEMINFOREQUEST._serialized_start=2529
  _QUERYSYSTEMINFOREQUEST._serialized_end=2553
  _QUERYSYSTEMINFORESPONSE._serialized_start=2555
  _QUERYSYSTEMINFORESPONSE._serialized_end=2625
  _CLOUDTPUDRIVER._serialized_start=2628
  _CLOUDTPUDRIVER._serialized_end=2995
# @@protoc_insertion_point(module_scope)
