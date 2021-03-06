from typing import List, Any, Type

import numpy as np

from .tensorflow_proto.tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2

_types_map = (
    (np.int32, types_pb2.DT_INT32), (np.int64, types_pb2.DT_INT64), (np.float32, types_pb2.DT_FLOAT),
    (np.float64, types_pb2.DT_DOUBLE), (np.bool_, types_pb2.DT_BOOL), (np.bytes_, types_pb2.DT_STRING)
)

_TF_TYPE_MAP = {tf_type: np_type for np_type, tf_type in _types_map}
_NP_TYPE_MAP = {np_type: tf_type for np_type, tf_type in _types_map}


def get_tf_type(np_dtype: Type):
    """
    :param np_type: python Type
    :return: types_pb2.DataType
    """
    try:
        return _NP_TYPE_MAP[np_dtype.type]
    except KeyError:
        raise TypeError(f'Could not infer tensorflow type for {np_dtype.type}')


def get_np_dtype(tf_type: types_pb2.DataType):
    """
    :param tf_type: types_pb2.DataType
    :return: types_pb2.DataType
    """
    try:
        return np.dtype(_TF_TYPE_MAP[tf_type])
    except KeyError:
        raise TypeError(f'Could not infer numpy type for {tf_type}')


def make_tensor_proto(values: List[Any]):
    np_array = np.asarray(values)

    # python/numpy default float type is float64. We prefer float32 instead.
    if np_array.dtype == np.float64:
        np_array = np_array.astype(np.float32)
    # python/numpy default int type is int64. We prefer int32 instead.
    elif np_array.dtype == np.int64:
        downcasted_array = np_array.astype(np.int32)
        # Do not down cast if it leads to precision loss.
        if np.array_equal(downcasted_array, np_array):
            np_array = downcasted_array

    dtype = get_tf_type(np_array.dtype)

    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=size) for size in np_array.shape]
    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)

    tensor_kwargs = {}
    if dtype == types_pb2.DT_STRING:
        string_val = []
        for vector in np_array:
            for s in vector:
                if not isinstance(s, bytes):
                    raise TypeError(f'{values} expect a list of bytes when working with DT_STRING types')
            string_val.append(s)
        tensor_kwargs['string_val'] = string_val
    else:
        tensor_kwargs['tensor_content'] = np_array.tobytes()
    return tensor_pb2.TensorProto(
        dtype=dtype,
        tensor_shape=tensor_shape_proto,
        **tensor_kwargs
    )


def make_ndarray_from_tensor(tensor: tensor_pb2.TensorProto):
    shape = [d.size for d in tensor.tensor_shape.dim]
    np_dtype = get_np_dtype(tensor.dtype)
    if tensor.tensor_content:
        return np.frombuffer(tensor.tensor_content, dtype=np_dtype).copy().reshape(shape)

    if tensor.dtype == types_pb2.DT_FLOAT:
        values = np.fromiter(tensor.float_val, dtype=np_dtype)
    elif tensor.dtype == types_pb2.DT_DOUBLE:
        values = np.fromiter(tensor.double_val, dtype=np_dtype)
    elif tensor.dtype in (types_pb2.DT_INT32, types_pb2.DT_INT64):
        values = np.fromiter(tensor.int_val, dtype=np_dtype)
    elif tensor.dtype == types_pb2.DT_BOOL:
        values = np.fromiter(tensor.bool_val, dtype=np_dtype)
    else:
        raise TypeError("Unsupported tensor type: %s" % tensor.dtype)

    if values.size == 0:
        return np.zeros(shape, np_dtype)

    num_elements = np.prod(shape, dtype=np.int64)
    if values.size != num_elements:
        values = np.pad(values, (0, num_elements - values.size), "edge")

    return values.reshape(shape)
