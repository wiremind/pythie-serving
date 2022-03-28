import os
import pickle
import logging

import grpc
import numpy as np

from xgboost import DMatrix

from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .tensorflow_proto.tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from .utils import make_ndarray_from_tensor
from .exceptions import PythieServingException


class XGBoostPredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer):

    def __init__(self, *, logger: logging.Logger, model_server_config: model_server_config_pb2.ModelServerConfig):
        self.logger = logger
        self.model_map = {}
        for model_config in model_server_config.model_config_list.config:
            with open(os.path.join(model_config.base_path, model_config.name) + ".pickled", 'rb') as opened_model:
                model = pickle.load(opened_model)
                model.set_param({'predictor': 'cpu_predictor'})
                self.model_map[model_config.name] = {'model': model, 'feature_names': model.feature_names}

    def Predict(self, request: predict_pb2.PredictRequest, context: grpc.RpcContext):
        model_name = request.model_spec.name
        if model_name not in self.model_map:
            raise PythieServingException(f'Unknown model: {model_name}. This pythie-serving instance can only '
                                         f'serve one of the following: {",".join(self.model_map.keys())}')

        model_dict = self.model_map[model_name]

        features_names = model_dict['feature_names']
        feature_rows = []
        for feature_name in features_names:
            if feature_name not in request.inputs:
                raise PythieServingException(f'{feature_name} not set in the predict request')
            nd_array = make_ndarray_from_tensor(request.inputs[feature_name])
            if len(nd_array.shape) != 2 or nd_array.shape[1] != 1:
                raise PythieServingException('All input vectors should be 1D tensor')
            feature_rows.append(nd_array)

        if len(set(len(l) for l in feature_rows)) != 1:
            raise PythieServingException('All input vectors should have the same length')

        model = model_dict['model']
        d_matrix = DMatrix(np.concatenate(feature_rows, axis=1), feature_names=features_names)
        outputs = model.predict(d_matrix, ntree_limit=model.best_ntree_limit)
        outputs = outputs.reshape((outputs.size, 1))  # return 1D tensor
        return outputs
