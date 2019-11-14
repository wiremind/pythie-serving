# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/lite/toco/model_flags.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from pythie_serving.tensorflow_proto.tensorflow.lite.toco import types_pb2 as tensorflow_dot_lite_dot_toco_dot_types__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/lite/toco/model_flags.proto',
  package='toco',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n&tensorflow/lite/toco/model_flags.proto\x12\x04toco\x1a tensorflow/lite/toco/types.proto\"\x1f\n\x0fInputArrayShape\x12\x0c\n\x04\x64ims\x18\x02 \x03(\x05\"\x8f\x01\n\nInputArray\x12\x0c\n\x04name\x18\x01 \x01(\t\x12$\n\x05shape\x18\x06 \x01(\x0b\x32\x15.toco.InputArrayShape\x12\x12\n\nmean_value\x18\x03 \x01(\x02\x12\x14\n\tstd_value\x18\x04 \x01(\x02:\x01\x31\x12#\n\tdata_type\x18\x05 \x01(\x0e\x32\x10.toco.IODataType\"t\n\x08RnnState\x12\x13\n\x0bstate_array\x18\x01 \x01(\t\x12\x1e\n\x16\x62\x61\x63k_edge_source_array\x18\x02 \x01(\t\x12\x13\n\x0b\x64iscardable\x18\x05 \x01(\x08\x12\x0c\n\x04size\x18\x03 \x01(\x05\x12\x10\n\x08num_dims\x18\x04 \x01(\x05\"\xef\x01\n\x0f\x41rraysExtraInfo\x12,\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x1b.toco.ArraysExtraInfo.Entry\x1a\xad\x01\n\x05\x45ntry\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0bname_regexp\x18\x07 \x01(\t\x12\x0b\n\x03min\x18\x02 \x01(\x01\x12\x0b\n\x03max\x18\x03 \x01(\x01\x12#\n\tdata_type\x18\x04 \x01(\x0e\x32\x10.toco.IODataType\x12$\n\x05shape\x18\x05 \x01(\x0b\x32\x15.toco.InputArrayShape\x12\x1c\n\x14\x63onstant_float_value\x18\x06 \x01(\x02\"\xad\x03\n\nModelFlags\x12&\n\x0cinput_arrays\x18\x01 \x03(\x0b\x32\x10.toco.InputArray\x12\x15\n\routput_arrays\x18\x02 \x03(\t\x12\x16\n\x0evariable_batch\x18\n \x01(\x08\x12\"\n\nrnn_states\x18\x0c \x03(\x0b\x32\x0e.toco.RnnState\x12\x31\n\x0cmodel_checks\x18\x0e \x03(\x0b\x32\x1b.toco.ModelFlags.ModelCheck\x12 \n\x18\x61llow_nonexistent_arrays\x18\x10 \x01(\x08\x12\x1d\n\x15\x61llow_nonascii_arrays\x18\x11 \x01(\x08\x12\x30\n\x11\x61rrays_extra_info\x18\x12 \x01(\x0b\x32\x15.toco.ArraysExtraInfo\x12(\n\x1a\x63hange_concat_input_ranges\x18\x13 \x01(\x08:\x04true\x1aT\n\nModelCheck\x12\x18\n\ncount_type\x18\x01 \x01(\t:\x04None\x12\x15\n\tcount_min\x18\x02 \x01(\x05:\x02-1\x12\x15\n\tcount_max\x18\x03 \x01(\x05:\x02-1')
  ,
  dependencies=[tensorflow_dot_lite_dot_toco_dot_types__pb2.DESCRIPTOR,])




_INPUTARRAYSHAPE = _descriptor.Descriptor(
  name='InputArrayShape',
  full_name='toco.InputArrayShape',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dims', full_name='toco.InputArrayShape.dims', index=0,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=82,
  serialized_end=113,
)


_INPUTARRAY = _descriptor.Descriptor(
  name='InputArray',
  full_name='toco.InputArray',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='toco.InputArray.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='toco.InputArray.shape', index=1,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean_value', full_name='toco.InputArray.mean_value', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='std_value', full_name='toco.InputArray.std_value', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_type', full_name='toco.InputArray.data_type', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=116,
  serialized_end=259,
)


_RNNSTATE = _descriptor.Descriptor(
  name='RnnState',
  full_name='toco.RnnState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state_array', full_name='toco.RnnState.state_array', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='back_edge_source_array', full_name='toco.RnnState.back_edge_source_array', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='discardable', full_name='toco.RnnState.discardable', index=2,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='size', full_name='toco.RnnState.size', index=3,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_dims', full_name='toco.RnnState.num_dims', index=4,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=261,
  serialized_end=377,
)


_ARRAYSEXTRAINFO_ENTRY = _descriptor.Descriptor(
  name='Entry',
  full_name='toco.ArraysExtraInfo.Entry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='toco.ArraysExtraInfo.Entry.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name_regexp', full_name='toco.ArraysExtraInfo.Entry.name_regexp', index=1,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min', full_name='toco.ArraysExtraInfo.Entry.min', index=2,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max', full_name='toco.ArraysExtraInfo.Entry.max', index=3,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_type', full_name='toco.ArraysExtraInfo.Entry.data_type', index=4,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='toco.ArraysExtraInfo.Entry.shape', index=5,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='constant_float_value', full_name='toco.ArraysExtraInfo.Entry.constant_float_value', index=6,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=446,
  serialized_end=619,
)

_ARRAYSEXTRAINFO = _descriptor.Descriptor(
  name='ArraysExtraInfo',
  full_name='toco.ArraysExtraInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='entries', full_name='toco.ArraysExtraInfo.entries', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_ARRAYSEXTRAINFO_ENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=380,
  serialized_end=619,
)


_MODELFLAGS_MODELCHECK = _descriptor.Descriptor(
  name='ModelCheck',
  full_name='toco.ModelFlags.ModelCheck',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='count_type', full_name='toco.ModelFlags.ModelCheck.count_type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("None").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='count_min', full_name='toco.ModelFlags.ModelCheck.count_min', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='count_max', full_name='toco.ModelFlags.ModelCheck.count_max', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
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
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=967,
  serialized_end=1051,
)

_MODELFLAGS = _descriptor.Descriptor(
  name='ModelFlags',
  full_name='toco.ModelFlags',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input_arrays', full_name='toco.ModelFlags.input_arrays', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_arrays', full_name='toco.ModelFlags.output_arrays', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='variable_batch', full_name='toco.ModelFlags.variable_batch', index=2,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rnn_states', full_name='toco.ModelFlags.rnn_states', index=3,
      number=12, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_checks', full_name='toco.ModelFlags.model_checks', index=4,
      number=14, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='allow_nonexistent_arrays', full_name='toco.ModelFlags.allow_nonexistent_arrays', index=5,
      number=16, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='allow_nonascii_arrays', full_name='toco.ModelFlags.allow_nonascii_arrays', index=6,
      number=17, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='arrays_extra_info', full_name='toco.ModelFlags.arrays_extra_info', index=7,
      number=18, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='change_concat_input_ranges', full_name='toco.ModelFlags.change_concat_input_ranges', index=8,
      number=19, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_MODELFLAGS_MODELCHECK, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=622,
  serialized_end=1051,
)

_INPUTARRAY.fields_by_name['shape'].message_type = _INPUTARRAYSHAPE
_INPUTARRAY.fields_by_name['data_type'].enum_type = tensorflow_dot_lite_dot_toco_dot_types__pb2._IODATATYPE
_ARRAYSEXTRAINFO_ENTRY.fields_by_name['data_type'].enum_type = tensorflow_dot_lite_dot_toco_dot_types__pb2._IODATATYPE
_ARRAYSEXTRAINFO_ENTRY.fields_by_name['shape'].message_type = _INPUTARRAYSHAPE
_ARRAYSEXTRAINFO_ENTRY.containing_type = _ARRAYSEXTRAINFO
_ARRAYSEXTRAINFO.fields_by_name['entries'].message_type = _ARRAYSEXTRAINFO_ENTRY
_MODELFLAGS_MODELCHECK.containing_type = _MODELFLAGS
_MODELFLAGS.fields_by_name['input_arrays'].message_type = _INPUTARRAY
_MODELFLAGS.fields_by_name['rnn_states'].message_type = _RNNSTATE
_MODELFLAGS.fields_by_name['model_checks'].message_type = _MODELFLAGS_MODELCHECK
_MODELFLAGS.fields_by_name['arrays_extra_info'].message_type = _ARRAYSEXTRAINFO
DESCRIPTOR.message_types_by_name['InputArrayShape'] = _INPUTARRAYSHAPE
DESCRIPTOR.message_types_by_name['InputArray'] = _INPUTARRAY
DESCRIPTOR.message_types_by_name['RnnState'] = _RNNSTATE
DESCRIPTOR.message_types_by_name['ArraysExtraInfo'] = _ARRAYSEXTRAINFO
DESCRIPTOR.message_types_by_name['ModelFlags'] = _MODELFLAGS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InputArrayShape = _reflection.GeneratedProtocolMessageType('InputArrayShape', (_message.Message,), {
  'DESCRIPTOR' : _INPUTARRAYSHAPE,
  '__module__' : 'tensorflow.lite.toco.model_flags_pb2'
  # @@protoc_insertion_point(class_scope:toco.InputArrayShape)
  })
_sym_db.RegisterMessage(InputArrayShape)

InputArray = _reflection.GeneratedProtocolMessageType('InputArray', (_message.Message,), {
  'DESCRIPTOR' : _INPUTARRAY,
  '__module__' : 'tensorflow.lite.toco.model_flags_pb2'
  # @@protoc_insertion_point(class_scope:toco.InputArray)
  })
_sym_db.RegisterMessage(InputArray)

RnnState = _reflection.GeneratedProtocolMessageType('RnnState', (_message.Message,), {
  'DESCRIPTOR' : _RNNSTATE,
  '__module__' : 'tensorflow.lite.toco.model_flags_pb2'
  # @@protoc_insertion_point(class_scope:toco.RnnState)
  })
_sym_db.RegisterMessage(RnnState)

ArraysExtraInfo = _reflection.GeneratedProtocolMessageType('ArraysExtraInfo', (_message.Message,), {

  'Entry' : _reflection.GeneratedProtocolMessageType('Entry', (_message.Message,), {
    'DESCRIPTOR' : _ARRAYSEXTRAINFO_ENTRY,
    '__module__' : 'tensorflow.lite.toco.model_flags_pb2'
    # @@protoc_insertion_point(class_scope:toco.ArraysExtraInfo.Entry)
    })
  ,
  'DESCRIPTOR' : _ARRAYSEXTRAINFO,
  '__module__' : 'tensorflow.lite.toco.model_flags_pb2'
  # @@protoc_insertion_point(class_scope:toco.ArraysExtraInfo)
  })
_sym_db.RegisterMessage(ArraysExtraInfo)
_sym_db.RegisterMessage(ArraysExtraInfo.Entry)

ModelFlags = _reflection.GeneratedProtocolMessageType('ModelFlags', (_message.Message,), {

  'ModelCheck' : _reflection.GeneratedProtocolMessageType('ModelCheck', (_message.Message,), {
    'DESCRIPTOR' : _MODELFLAGS_MODELCHECK,
    '__module__' : 'tensorflow.lite.toco.model_flags_pb2'
    # @@protoc_insertion_point(class_scope:toco.ModelFlags.ModelCheck)
    })
  ,
  'DESCRIPTOR' : _MODELFLAGS,
  '__module__' : 'tensorflow.lite.toco.model_flags_pb2'
  # @@protoc_insertion_point(class_scope:toco.ModelFlags)
  })
_sym_db.RegisterMessage(ModelFlags)
_sym_db.RegisterMessage(ModelFlags.ModelCheck)


# @@protoc_insertion_point(module_scope)
