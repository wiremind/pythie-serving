# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/data/service/journal.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.core.data.service import common_pb2 as tensorflow_dot_core_dot_data_dot_service_dot_common__pb2
from pythie_serving.tensorflow_proto.tensorflow.core.protobuf import data_service_pb2 as tensorflow_dot_core_dot_protobuf_dot_data__service__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*tensorflow/core/data/service/journal.proto\x12\x0ftensorflow.data\x1a)tensorflow/core/data/service/common.proto\x1a+tensorflow/core/protobuf/data_service.proto\"\x93\x07\n\x06Update\x12\x42\n\x10register_dataset\x18\x01 \x01(\x0b\x32&.tensorflow.data.RegisterDatasetUpdateH\x00\x12@\n\x0fregister_worker\x18\x05 \x01(\x0b\x32%.tensorflow.data.RegisterWorkerUpdateH\x00\x12\x36\n\ncreate_job\x18\x0e \x01(\x0b\x32 .tensorflow.data.CreateJobUpdateH\x00\x12\x42\n\x10\x63reate_iteration\x18\x02 \x01(\x0b\x32&.tensorflow.data.CreateIterationUpdateH\x00\x12<\n\rproduce_split\x18\x08 \x01(\x0b\x32#.tensorflow.data.ProduceSplitUpdateH\x00\x12Q\n\x18\x61\x63quire_iteration_client\x18\x06 \x01(\x0b\x32-.tensorflow.data.AcquireIterationClientUpdateH\x00\x12Q\n\x18release_iteration_client\x18\x07 \x01(\x0b\x32-.tensorflow.data.ReleaseIterationClientUpdateH\x00\x12S\n\x19garbage_collect_iteration\x18\x0c \x01(\x0b\x32..tensorflow.data.GarbageCollectIterationUpdateH\x00\x12\x38\n\x0bremove_task\x18\x0b \x01(\x0b\x32!.tensorflow.data.RemoveTaskUpdateH\x00\x12G\n\x13\x63reate_pending_task\x18\t \x01(\x0b\x32(.tensorflow.data.CreatePendingTaskUpdateH\x00\x12\x42\n\x10\x63lient_heartbeat\x18\n \x01(\x0b\x32&.tensorflow.data.ClientHeartbeatUpdateH\x00\x12\x38\n\x0b\x63reate_task\x18\x03 \x01(\x0b\x32!.tensorflow.data.CreateTaskUpdateH\x00\x12\x38\n\x0b\x66inish_task\x18\x04 \x01(\x0b\x32!.tensorflow.data.FinishTaskUpdateH\x00\x42\r\n\x0bupdate_typeJ\x04\x08\r\x10\x0e\"\x96\x01\n\x15RegisterDatasetUpdate\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x13\n\x0b\x66ingerprint\x18\x02 \x01(\x04\x12\x36\n\x08metadata\x18\x03 \x01(\x0b\x32$.tensorflow.data.DataServiceMetadata\x12\x1c\n\x14\x64\x65\x64upe_by_dataset_id\x18\x04 \x01(\x08\"q\n\x14RegisterWorkerUpdate\x12\x16\n\x0eworker_address\x18\x01 \x01(\t\x12\x18\n\x10transfer_address\x18\x02 \x01(\t\x12\x13\n\x0bworker_tags\x18\x03 \x03(\t\x12\x12\n\nworker_uid\x18\x04 \x01(\x03\"\x94\x02\n\x0f\x43reateJobUpdate\x12\x0e\n\x06job_id\x18\x01 \x01(\x03\x12\x10\n\x08job_name\x18\x02 \x01(\t\x12\x12\n\ndataset_id\x18\x03 \x01(\t\x12?\n\x13processing_mode_def\x18\x04 \x01(\x0b\x32\".tensorflow.data.ProcessingModeDef\x12\x17\n\rnum_consumers\x18\x06 \x01(\x03H\x00\x12\x36\n\x0etarget_workers\x18\x07 \x01(\x0e\x32\x1e.tensorflow.data.TargetWorkers\x12\x1f\n\x17use_cross_trainer_cache\x18\x08 \x01(\x08\x42\x18\n\x16optional_num_consumers\"n\n\x15\x43reateIterationUpdate\x12\x14\n\x0citeration_id\x18\x01 \x01(\x03\x12\x0e\n\x06job_id\x18\x02 \x01(\x03\x12\x12\n\nrepetition\x18\x03 \x01(\x03\x12\x1b\n\x13num_split_providers\x18\x04 \x01(\x03\"n\n\x12ProduceSplitUpdate\x12\x14\n\x0citeration_id\x18\x01 \x01(\x03\x12\x12\n\nrepetition\x18\x02 \x01(\x03\x12\x1c\n\x14split_provider_index\x18\x04 \x01(\x03\x12\x10\n\x08\x66inished\x18\x03 \x01(\x08\"Q\n\x1c\x41\x63quireIterationClientUpdate\x12\x14\n\x0citeration_id\x18\x01 \x01(\x03\x12\x1b\n\x13iteration_client_id\x18\x02 \x01(\x03\"P\n\x1cReleaseIterationClientUpdate\x12\x1b\n\x13iteration_client_id\x18\x01 \x01(\x03\x12\x13\n\x0btime_micros\x18\x02 \x01(\x03\"5\n\x1dGarbageCollectIterationUpdate\x12\x14\n\x0citeration_id\x18\x01 \x01(\x03\"#\n\x10RemoveTaskUpdate\x12\x0f\n\x07task_id\x18\x01 \x01(\x03\"(\n\x0cTaskRejected\x12\x18\n\x10new_target_round\x18\x01 \x01(\x03\"\x81\x01\n\x15\x43lientHeartbeatUpdate\x12\x1b\n\x13iteration_client_id\x18\x01 \x01(\x03\x12\x15\n\rtask_accepted\x18\x02 \x01(\x08\x12\x34\n\rtask_rejected\x18\x03 \x01(\x0b\x32\x1d.tensorflow.data.TaskRejected\"\xb3\x01\n\x17\x43reatePendingTaskUpdate\x12\x0f\n\x07task_id\x18\x01 \x01(\x03\x12\x14\n\x0citeration_id\x18\x02 \x01(\x03\x12\x16\n\x0eworker_address\x18\x03 \x01(\t\x12\x18\n\x10transfer_address\x18\x04 \x01(\t\x12\x13\n\x0bworker_tags\x18\x06 \x03(\t\x12\x12\n\nworker_uid\x18\x07 \x01(\x03\x12\x16\n\x0estarting_round\x18\x05 \x01(\x03\"\xa0\x01\n\x10\x43reateTaskUpdate\x12\x0f\n\x07task_id\x18\x01 \x01(\x03\x12\x14\n\x0citeration_id\x18\x02 \x01(\x03\x12\x16\n\x0eworker_address\x18\x04 \x01(\t\x12\x18\n\x10transfer_address\x18\x06 \x01(\t\x12\x13\n\x0bworker_tags\x18\x07 \x03(\t\x12\x12\n\nworker_uid\x18\x08 \x01(\x03J\x04\x08\x03\x10\x04J\x04\x08\x05\x10\x06\"#\n\x10\x46inishTaskUpdate\x12\x0f\n\x07task_id\x18\x01 \x01(\x03\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tensorflow.core.data.service.journal_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _UPDATE._serialized_start=152
  _UPDATE._serialized_end=1067
  _REGISTERDATASETUPDATE._serialized_start=1070
  _REGISTERDATASETUPDATE._serialized_end=1220
  _REGISTERWORKERUPDATE._serialized_start=1222
  _REGISTERWORKERUPDATE._serialized_end=1335
  _CREATEJOBUPDATE._serialized_start=1338
  _CREATEJOBUPDATE._serialized_end=1614
  _CREATEITERATIONUPDATE._serialized_start=1616
  _CREATEITERATIONUPDATE._serialized_end=1726
  _PRODUCESPLITUPDATE._serialized_start=1728
  _PRODUCESPLITUPDATE._serialized_end=1838
  _ACQUIREITERATIONCLIENTUPDATE._serialized_start=1840
  _ACQUIREITERATIONCLIENTUPDATE._serialized_end=1921
  _RELEASEITERATIONCLIENTUPDATE._serialized_start=1923
  _RELEASEITERATIONCLIENTUPDATE._serialized_end=2003
  _GARBAGECOLLECTITERATIONUPDATE._serialized_start=2005
  _GARBAGECOLLECTITERATIONUPDATE._serialized_end=2058
  _REMOVETASKUPDATE._serialized_start=2060
  _REMOVETASKUPDATE._serialized_end=2095
  _TASKREJECTED._serialized_start=2097
  _TASKREJECTED._serialized_end=2137
  _CLIENTHEARTBEATUPDATE._serialized_start=2140
  _CLIENTHEARTBEATUPDATE._serialized_end=2269
  _CREATEPENDINGTASKUPDATE._serialized_start=2272
  _CREATEPENDINGTASKUPDATE._serialized_end=2451
  _CREATETASKUPDATE._serialized_start=2454
  _CREATETASKUPDATE._serialized_end=2614
  _FINISHTASKUPDATE._serialized_start=2616
  _FINISHTASKUPDATE._serialized_end=2651
# @@protoc_insertion_point(module_scope)
