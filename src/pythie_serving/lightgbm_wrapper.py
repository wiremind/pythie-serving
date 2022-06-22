import logging
import os
import pickle

import grpc
from lightgbm import Booster

from .exceptions import PythieServingException
from .tensorflow_proto.tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc,
)
from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .utils import (
    check_array_shape,
    check_request_feature_exists,
    make_ndarray_from_tensor,
)


class LightGBMPredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer):
    def __init__(
        self,
        *,
        logger: logging.Logger,
        model_server_config: model_server_config_pb2.ModelServerConfig,
    ):
        self.logger = logger
        self.model_map = {}
        for model_config in model_server_config.model_config_list.config:
            with open(
                os.path.join(model_config.base_path, model_config.name) + ".pickled",
                "rb",
            ) as opened_model:
                model = pickle.load(opened_model)

                if isinstance(model, Booster):
                    feature_names = model.feature_name()
                    best_iteration = model.best_iteration
                else:
                    feature_names = model.feature_names
                    best_iteration = getattr(model, "best_iteration", None)

                self.model_map[model_config.name] = {
                    "model": model,
                    "feature_names": feature_names,
                    "best_iteration": best_iteration,
                }

    def Predict(self, request: predict_pb2.PredictRequest, context: grpc.RpcContext):
        model_name = request.model_spec.name
        if model_name not in self.model_map:
            raise PythieServingException(
                f"Unknown model: {model_name}. This pythie-serving instance can only "
                f'serve one of the following: {",".join(self.model_map.keys())}'
            )

        model_dict = self.model_map[model_name]

        features_names = model_dict["feature_names"]
        samples = None
        for feature_name in features_names:
            check_request_feature_exists(request.inputs, feature_name)

            nd_array = make_ndarray_from_tensor(request.inputs[feature_name])
            check_array_shape(nd_array)

            if samples is None:
                samples = [[] for _ in range(nd_array.shape[0])]

            for sample_index, value in enumerate(nd_array):
                samples[sample_index].append(value[0])

        model = model_dict["model"]
        kwargs = {}
        if model_dict["best_iteration"]:
            kwargs["best_iteration"] = model_dict["best_iteration"]
        return model.predict(samples, **kwargs)
