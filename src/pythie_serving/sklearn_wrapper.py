import os
import pickle
import logging

import grpc
import numpy as np
# import pandas as pd
import json

from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .tensorflow_proto.tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from .utils import make_ndarray_from_tensor
from .exceptions import PythieServingException


class SklearnPredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer):

    def __init__(self, *, logger: logging.Logger, model_server_config: model_server_config_pb2.ModelServerConfig):
        self.logger = logger
        self.model_map = {}
        for model_config in model_server_config.model_config_list.config:
            with open(os.path.join(model_config.base_path, model_config.name) + ".pickled", 'rb') as opened_model:
                model = pickle.load(opened_model)
                self.model_map[model_config.name] = {'model': model}

            with open(os.path.join(model_config.base_path, "metadata.json"), "r") as f:
                metadata = json.load(f)

            self.model_map[model_config.name] = {
                "model": model,
                "feature_names": metadata["feature_names"],
                "nb_features": len(metadata["feature_names"])
            }

    def Predict(self, request: predict_pb2.PredictRequest, context: grpc.RpcContext):
        model_name = request.model_spec.name
        if model_name not in self.model_map:
            raise PythieServingException(f'Unknown model: {model_name}. This pythie-serving instance can only '
                                         f'serve one of the following: {",".join(self.model_map.keys())}')

        model_dict = self.model_map[model_name]
        model = model_dict["model"]
        features_names = model_dict["feature_names"]
        nb_features = model_dict["nb_features"]

        for feature_name in features_names:
            if feature_name not in request.inputs:
                raise PythieServingException(f"{feature_name} not set in the predict request.")

        nb_samples = request.inputs[features_names[0]].tensor_shape.dim[0].size
        samples = np.empty((nb_samples, nb_features), object)

        for feature_index, feature_name in enumerate(features_names):

            if request.inputs[features_names[0]].tensor_shape.dim[0].size != nb_samples:
                raise PythieServingException(f"{feature_name} has invalid length.")

            nd_array = make_ndarray_from_tensor(request.inputs[feature_name])
            if len(nd_array.shape) != 2 or nd_array.shape[1] != 1:
                raise PythieServingException("All input vectors should be 1D tensor")

            samples[:, feature_index] = nd_array.reshape(-1)

        return model.predict(samples)
