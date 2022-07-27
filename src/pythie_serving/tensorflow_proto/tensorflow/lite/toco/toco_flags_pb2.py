# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/lite/toco/toco_flags.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.lite.toco import types_pb2 as tensorflow_dot_lite_dot_toco_dot_types__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n%tensorflow/lite/toco/toco_flags.proto\x12\x04toco\x1a tensorflow/lite/toco/types.proto\"\xd4\t\n\tTocoFlags\x12&\n\x0cinput_format\x18\x01 \x01(\x0e\x32\x10.toco.FileFormat\x12\'\n\routput_format\x18\x02 \x01(\x0e\x32\x10.toco.FileFormat\x12.\n\x14inference_input_type\x18\x0b \x01(\x0e\x32\x10.toco.IODataType\x12(\n\x0einference_type\x18\x04 \x01(\x0e\x32\x10.toco.IODataType\x12\x1a\n\x12\x64\x65\x66\x61ult_ranges_min\x18\x05 \x01(\x02\x12\x1a\n\x12\x64\x65\x66\x61ult_ranges_max\x18\x06 \x01(\x02\x12 \n\x18\x64\x65\x66\x61ult_int16_ranges_min\x18\x0f \x01(\x02\x12 \n\x18\x64\x65\x66\x61ult_int16_ranges_max\x18\x10 \x01(\x02\x12\x17\n\x0f\x64rop_fake_quant\x18\x07 \x01(\x08\x12!\n\x19reorder_across_fake_quant\x18\x08 \x01(\x08\x12\x18\n\x10\x61llow_custom_ops\x18\n \x01(\x08\x12\x1f\n\x17\x64rop_control_dependency\x18\x0c \x01(\x08\x12+\n#debug_disable_recurrent_cell_fusion\x18\r \x01(\x08\x12%\n\x1dpropagate_fake_quant_num_bits\x18\x0e \x01(\x08\x12\x35\n-allow_nudging_weights_to_use_fast_gemm_kernel\x18\x11 \x01(\x08\x12\'\n\x1b\x64\x65\x64upe_array_min_size_bytes\x18\x12 \x01(\x03:\x02\x36\x34\x12&\n\x18split_tflite_lstm_inputs\x18\x13 \x01(\x08:\x04true\x12\x1f\n\x10quantize_weights\x18\x14 \x01(\x08:\x05\x66\x61lse\x12\x19\n\x11\x64ump_graphviz_dir\x18\x18 \x01(\t\x12#\n\x1b\x64ump_graphviz_include_video\x18\x19 \x01(\x08\x12%\n\x16post_training_quantize\x18\x1a \x01(\x08:\x05\x66\x61lse\x12#\n\x14\x65nable_select_tf_ops\x18\x1b \x01(\x08:\x05\x66\x61lse\x12\"\n\x13\x66orce_select_tf_ops\x18\x1c \x01(\x08:\x05\x66\x61lse\x12\"\n\x13quantize_to_float16\x18\x1d \x01(\x08:\x05\x66\x61lse\x12#\n\x15\x61llow_dynamic_tensors\x18\x1e \x01(\x08:\x04true\x12\x1e\n\x16\x63onversion_summary_dir\x18\x1f \x01(\t\x12\x19\n\rcustom_opdefs\x18  \x03(\tB\x02\x18\x01\x12\x1a\n\x12select_user_tf_ops\x18! \x03(\t\x12(\n enable_tflite_resource_variables\x18\" \x01(\x08\x12 \n\x12unfold_batchmatmul\x18# \x01(\x08:\x04true\x12#\n\x15lower_tensor_list_ops\x18$ \x01(\x08:\x04true\x12+\n\x11\x61\x63\x63umulation_type\x18% \x01(\x0e\x32\x10.toco.IODataType\x12\x1d\n\x0e\x61llow_bfloat16\x18& \x01(\x08:\x05\x66\x61lse\x12\x1f\n\x17\x61llow_all_select_tf_ops\x18\' \x01(\x08*\\\n\nFileFormat\x12\x17\n\x13\x46ILE_FORMAT_UNKNOWN\x10\x00\x12\x17\n\x13TENSORFLOW_GRAPHDEF\x10\x01\x12\n\n\x06TFLITE\x10\x02\x12\x10\n\x0cGRAPHVIZ_DOT\x10\x03')

_FILEFORMAT = DESCRIPTOR.enum_types_by_name['FileFormat']
FileFormat = enum_type_wrapper.EnumTypeWrapper(_FILEFORMAT)
FILE_FORMAT_UNKNOWN = 0
TENSORFLOW_GRAPHDEF = 1
TFLITE = 2
GRAPHVIZ_DOT = 3


_TOCOFLAGS = DESCRIPTOR.message_types_by_name['TocoFlags']
TocoFlags = _reflection.GeneratedProtocolMessageType('TocoFlags', (_message.Message,), {
  'DESCRIPTOR' : _TOCOFLAGS,
  '__module__' : 'tensorflow.lite.toco.toco_flags_pb2'
  # @@protoc_insertion_point(class_scope:toco.TocoFlags)
  })
_sym_db.RegisterMessage(TocoFlags)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TOCOFLAGS.fields_by_name['custom_opdefs']._options = None
  _TOCOFLAGS.fields_by_name['custom_opdefs']._serialized_options = b'\030\001'
  _FILEFORMAT._serialized_start=1320
  _FILEFORMAT._serialized_end=1412
  _TOCOFLAGS._serialized_start=82
  _TOCOFLAGS._serialized_end=1318
# @@protoc_insertion_point(module_scope)
