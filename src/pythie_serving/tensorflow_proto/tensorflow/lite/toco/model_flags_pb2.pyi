"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import tensorflow.lite.toco.types_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class InputArrayShape(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    DIMS_FIELD_NUMBER: builtins.int
    UNKNOWN_RANK_FIELD_NUMBER: builtins.int
    @property
    def dims(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Dimensions of the tensor."""
        pass
    unknown_rank: builtins.bool
    """If true, the number of dimensions in the shape is unknown.

    If true, "dims.size()" must be 0.
    """

    def __init__(self,
        *,
        dims: typing.Optional[typing.Iterable[builtins.int]] = ...,
        unknown_rank: typing.Optional[builtins.bool] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["unknown_rank",b"unknown_rank"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["dims",b"dims","unknown_rank",b"unknown_rank"]) -> None: ...
global___InputArrayShape = InputArrayShape

class InputArray(google.protobuf.message.Message):
    """Next ID to USE: 7."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NAME_FIELD_NUMBER: builtins.int
    SHAPE_FIELD_NUMBER: builtins.int
    MEAN_VALUE_FIELD_NUMBER: builtins.int
    STD_VALUE_FIELD_NUMBER: builtins.int
    DATA_TYPE_FIELD_NUMBER: builtins.int
    name: typing.Text
    """Name of the input arrays, i.e. the arrays from which input activations
    will be read.
    """

    @property
    def shape(self) -> global___InputArrayShape:
        """Shape of the input.  For many applications the dimensions are {batch,
        height, width, depth}.  Often the batch is left "unspecified" by providing
        a value of -1.

        The last dimension is typically called 'depth' or 'channels'. For example,
        for an image model taking RGB images as input, this would have the value 3.
        """
        pass
    mean_value: builtins.float
    """mean_value and std_value parameters control the interpretation of raw input
    activation values (elements of the input array) as real numbers. The
    mapping is given by:

       real_value = (raw_input_value - mean_value) / std_value

    In particular, the defaults (mean_value=0, std_value=1) yield
    real_value = raw_input_value. Often, non-default values are used in image
    models. For example, an image model taking uint8 image channel values as
    its raw inputs, in [0, 255] range, may use mean_value=128, std_value=128 to
    map them into the interval [-1, 1).

    Note: this matches exactly the meaning of mean_value and std_value in
    (TensorFlow via LegacyFedInput).
    """

    std_value: builtins.float
    data_type: tensorflow.lite.toco.types_pb2.IODataType.ValueType
    """Data type of the input.

    In many graphs, the input arrays already have defined data types,
    e.g. Placeholder nodes in a TensorFlow GraphDef have a dtype attribute.
    In those cases, it is not needed to specify this data_type flag.
    The purpose of this flag is only to define the data type of input
    arrays whose type isn't defined in the input graph file. For example,
    when specifying an arbitrary (not Placeholder) --input_array into
    a TensorFlow GraphDef.

    When this data_type is quantized (e.g. QUANTIZED_UINT8), the
    corresponding quantization parameters are the mean_value, std_value
    fields.

    It is also important to understand the nuance between this data_type
    flag and the inference_input_type in TocoFlags. The basic difference
    is that this data_type (like all ModelFlags) describes a property
    of the input graph, while inference_input_type (like all TocoFlags)
    describes an aspect of the toco transformation process and thus of
    the output file. The types of input arrays may be different between
    the input and output files if quantization or dequantization occurred.
    Such differences can only occur for real-number data i.e. only
    between FLOAT and quantized types (e.g. QUANTIZED_UINT8).
    """

    def __init__(self,
        *,
        name: typing.Optional[typing.Text] = ...,
        shape: typing.Optional[global___InputArrayShape] = ...,
        mean_value: typing.Optional[builtins.float] = ...,
        std_value: typing.Optional[builtins.float] = ...,
        data_type: typing.Optional[tensorflow.lite.toco.types_pb2.IODataType.ValueType] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["data_type",b"data_type","mean_value",b"mean_value","name",b"name","shape",b"shape","std_value",b"std_value"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["data_type",b"data_type","mean_value",b"mean_value","name",b"name","shape",b"shape","std_value",b"std_value"]) -> None: ...
global___InputArray = InputArray

class RnnState(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    STATE_ARRAY_FIELD_NUMBER: builtins.int
    BACK_EDGE_SOURCE_ARRAY_FIELD_NUMBER: builtins.int
    DISCARDABLE_FIELD_NUMBER: builtins.int
    SIZE_FIELD_NUMBER: builtins.int
    NUM_DIMS_FIELD_NUMBER: builtins.int
    state_array: typing.Text
    back_edge_source_array: typing.Text
    discardable: builtins.bool
    size: builtins.int
    """size allows to specify a 1-D shape for the RNN state array.
    Will be expanded with 1's to fit the model.
    TODO(benoitjacob): should allow a generic, explicit shape.
    """

    num_dims: builtins.int
    def __init__(self,
        *,
        state_array: typing.Optional[typing.Text] = ...,
        back_edge_source_array: typing.Optional[typing.Text] = ...,
        discardable: typing.Optional[builtins.bool] = ...,
        size: typing.Optional[builtins.int] = ...,
        num_dims: typing.Optional[builtins.int] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["back_edge_source_array",b"back_edge_source_array","discardable",b"discardable","num_dims",b"num_dims","size",b"size","state_array",b"state_array"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["back_edge_source_array",b"back_edge_source_array","discardable",b"discardable","num_dims",b"num_dims","size",b"size","state_array",b"state_array"]) -> None: ...
global___RnnState = RnnState

class ArraysExtraInfo(google.protobuf.message.Message):
    """An ArraysExtraInfo message stores a collection of additional Information
    about arrays in a model, complementing the information in the model itself.
    It is intentionally a separate message so that it may be serialized and
    passed separately from the model. See --arrays_extra_info_file.

    A typical use case is to manually specify MinMax for specific arrays in a
    model that does not already contain such MinMax information.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class Entry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        NAME_FIELD_NUMBER: builtins.int
        NAME_REGEXP_FIELD_NUMBER: builtins.int
        MIN_FIELD_NUMBER: builtins.int
        MAX_FIELD_NUMBER: builtins.int
        DATA_TYPE_FIELD_NUMBER: builtins.int
        SHAPE_FIELD_NUMBER: builtins.int
        CONSTANT_FLOAT_VALUE_FIELD_NUMBER: builtins.int
        name: typing.Text
        """Next ID to use: 8."""

        name_regexp: typing.Text
        min: builtins.float
        max: builtins.float
        data_type: tensorflow.lite.toco.types_pb2.IODataType.ValueType
        @property
        def shape(self) -> global___InputArrayShape: ...
        constant_float_value: builtins.float
        def __init__(self,
            *,
            name: typing.Optional[typing.Text] = ...,
            name_regexp: typing.Optional[typing.Text] = ...,
            min: typing.Optional[builtins.float] = ...,
            max: typing.Optional[builtins.float] = ...,
            data_type: typing.Optional[tensorflow.lite.toco.types_pb2.IODataType.ValueType] = ...,
            shape: typing.Optional[global___InputArrayShape] = ...,
            constant_float_value: typing.Optional[builtins.float] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["constant_float_value",b"constant_float_value","data_type",b"data_type","max",b"max","min",b"min","name",b"name","name_regexp",b"name_regexp","shape",b"shape"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["constant_float_value",b"constant_float_value","data_type",b"data_type","max",b"max","min",b"min","name",b"name","name_regexp",b"name_regexp","shape",b"shape"]) -> None: ...

    ENTRIES_FIELD_NUMBER: builtins.int
    @property
    def entries(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ArraysExtraInfo.Entry]: ...
    def __init__(self,
        *,
        entries: typing.Optional[typing.Iterable[global___ArraysExtraInfo.Entry]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["entries",b"entries"]) -> None: ...
global___ArraysExtraInfo = ArraysExtraInfo

class ModelFlags(google.protobuf.message.Message):
    """ModelFlags encodes properties of a model that, depending on the file
    format, may or may not be recorded in the model file. The purpose of
    representing these properties in ModelFlags is to allow passing them
    separately from the input model file, for instance as command-line
    parameters, so that we can offer a single uniform interface that can
    handle files from different input formats.

    For each of these properties, and each supported file format, we
    detail in comments below whether the property exists in the given file
    format.

    Obsolete flags that have been removed:
      optional int32 input_depth = 3;
      optional int32 input_width = 4;
      optional int32 input_height = 5;
      optional int32 batch = 6 [ default = 1];
      optional float mean_value = 7;
      optional float std_value = 8 [default = 1.];
      optional int32 input_dims = 11 [ default = 4];
      repeated int32 input_shape = 13;

    Next ID to USE: 25.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class ModelCheck(google.protobuf.message.Message):
        """Checks applied to the model, typically after toco's comprehensive
        graph transformations.
        Next ID to USE: 4.
        """
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        COUNT_TYPE_FIELD_NUMBER: builtins.int
        COUNT_MIN_FIELD_NUMBER: builtins.int
        COUNT_MAX_FIELD_NUMBER: builtins.int
        count_type: typing.Text
        """Use the name of a type of operator to check its counts.
        Use "Total" for overall operator counts.
        Use "Arrays" for overall array counts.
        """

        count_min: builtins.int
        """A count of zero is a meaningful check, so negative used to mean disable."""

        count_max: builtins.int
        """If count_max < count_min, then count_min is only allowed value."""

        def __init__(self,
            *,
            count_type: typing.Optional[typing.Text] = ...,
            count_min: typing.Optional[builtins.int] = ...,
            count_max: typing.Optional[builtins.int] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["count_max",b"count_max","count_min",b"count_min","count_type",b"count_type"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["count_max",b"count_max","count_min",b"count_min","count_type",b"count_type"]) -> None: ...

    INPUT_ARRAYS_FIELD_NUMBER: builtins.int
    OUTPUT_ARRAYS_FIELD_NUMBER: builtins.int
    CONTROL_OUTPUT_ARRAYS_FIELD_NUMBER: builtins.int
    VARIABLE_BATCH_FIELD_NUMBER: builtins.int
    RNN_STATES_FIELD_NUMBER: builtins.int
    MODEL_CHECKS_FIELD_NUMBER: builtins.int
    ALLOW_NONEXISTENT_ARRAYS_FIELD_NUMBER: builtins.int
    ALLOW_NONASCII_ARRAYS_FIELD_NUMBER: builtins.int
    ARRAYS_EXTRA_INFO_FIELD_NUMBER: builtins.int
    CHANGE_CONCAT_INPUT_RANGES_FIELD_NUMBER: builtins.int
    SAVED_MODEL_DIR_FIELD_NUMBER: builtins.int
    SAVED_MODEL_VERSION_FIELD_NUMBER: builtins.int
    SAVED_MODEL_TAGS_FIELD_NUMBER: builtins.int
    SAVED_MODEL_EXPORTED_NAMES_FIELD_NUMBER: builtins.int
    @property
    def input_arrays(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___InputArray]:
        """Information about the input arrays, i.e. the arrays from which input
        activations will be read.
        """
        pass
    @property
    def output_arrays(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """Name of the output arrays, i.e. the arrays into which output activations
        will be written.
        """
        pass
    @property
    def control_output_arrays(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """Name of the control outputs."""
        pass
    variable_batch: builtins.bool
    """If true, the model accepts an arbitrary batch size. Mutually exclusive with
    the 'batch' field: at most one of these two fields can be set.
    """

    @property
    def rnn_states(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RnnState]: ...
    @property
    def model_checks(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ModelFlags.ModelCheck]: ...
    allow_nonexistent_arrays: builtins.bool
    """If true, will allow passing inexistent arrays in --input_arrays
    and --output_arrays. This makes little sense, is only useful to
    more easily get graph visualizations.
    """

    allow_nonascii_arrays: builtins.bool
    """If true, will allow passing non-ascii-printable characters in
    --input_arrays and --output_arrays. By default (if false), only
    ascii printable characters are allowed, i.e. character codes
    ranging from 32 to 127. This is disallowed by default so as to
    catch common copy-and-paste issues where invisible unicode
    characters are unwittingly added to these strings.
    """

    @property
    def arrays_extra_info(self) -> global___ArraysExtraInfo:
        """If set, this ArraysExtraInfo allows to pass extra information about arrays
        not specified in the input model file, such as extra MinMax information.
        """
        pass
    change_concat_input_ranges: builtins.bool
    """When set to false, toco will not change the input ranges and the output
    ranges of concat operator to the overlap of all input ranges.
    """

    saved_model_dir: typing.Text
    """Filepath of the saved model to be converted. This value will be non-empty
    only when the saved model import path will be used. Otherwise, the graph
    def-based conversion will be processed.
    """

    saved_model_version: builtins.int
    """SavedModel file format version of The saved model file to be converted.
    This value will be set only when the SavedModel import path will be used.
    """

    @property
    def saved_model_tags(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """Set of string saved model tags, formatted in the comma-separated value.
        This value will be set only when the SavedModel import path will be used.
        """
        pass
    @property
    def saved_model_exported_names(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """Names to be exported (default: export all) when the saved model import path
        is on. This value will be set only when the SavedModel import path will be
        used.
        """
        pass
    def __init__(self,
        *,
        input_arrays: typing.Optional[typing.Iterable[global___InputArray]] = ...,
        output_arrays: typing.Optional[typing.Iterable[typing.Text]] = ...,
        control_output_arrays: typing.Optional[typing.Iterable[typing.Text]] = ...,
        variable_batch: typing.Optional[builtins.bool] = ...,
        rnn_states: typing.Optional[typing.Iterable[global___RnnState]] = ...,
        model_checks: typing.Optional[typing.Iterable[global___ModelFlags.ModelCheck]] = ...,
        allow_nonexistent_arrays: typing.Optional[builtins.bool] = ...,
        allow_nonascii_arrays: typing.Optional[builtins.bool] = ...,
        arrays_extra_info: typing.Optional[global___ArraysExtraInfo] = ...,
        change_concat_input_ranges: typing.Optional[builtins.bool] = ...,
        saved_model_dir: typing.Optional[typing.Text] = ...,
        saved_model_version: typing.Optional[builtins.int] = ...,
        saved_model_tags: typing.Optional[typing.Iterable[typing.Text]] = ...,
        saved_model_exported_names: typing.Optional[typing.Iterable[typing.Text]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["allow_nonascii_arrays",b"allow_nonascii_arrays","allow_nonexistent_arrays",b"allow_nonexistent_arrays","arrays_extra_info",b"arrays_extra_info","change_concat_input_ranges",b"change_concat_input_ranges","saved_model_dir",b"saved_model_dir","saved_model_version",b"saved_model_version","variable_batch",b"variable_batch"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["allow_nonascii_arrays",b"allow_nonascii_arrays","allow_nonexistent_arrays",b"allow_nonexistent_arrays","arrays_extra_info",b"arrays_extra_info","change_concat_input_ranges",b"change_concat_input_ranges","control_output_arrays",b"control_output_arrays","input_arrays",b"input_arrays","model_checks",b"model_checks","output_arrays",b"output_arrays","rnn_states",b"rnn_states","saved_model_dir",b"saved_model_dir","saved_model_exported_names",b"saved_model_exported_names","saved_model_tags",b"saved_model_tags","saved_model_version",b"saved_model_version","variable_batch",b"variable_batch"]) -> None: ...
global___ModelFlags = ModelFlags
