import json
import logging
import os

import grpc
from treelite_runtime import DMatrix, Predictor

from .exceptions import PythieServingException
from .tensorflow_proto.tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc,
)
from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .utils import parse_sample


class TreelitePredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer):
    def __init__(
        self,
        *,
        logger: logging.Logger,
        model_server_config: model_server_config_pb2.ModelServerConfig,
    ):
        self.logger = logger
        self.model_map = {}
        for model_config in model_server_config.model_config_list.config:
            model = Predictor(
                libpath=os.path.join(model_config.base_path, model_config.name) + ".so",
                nthread=int(os.environ.get("TREELITE_NTHREAD", 1)),
            )

            with open(os.path.join(model_config.base_path, "metadata.json")) as f:
                metadata = json.load(f)

            self.model_map[model_config.name] = {
                "model": model,
                "feature_names": metadata["feature_names"],
                "nb_features": len(metadata["feature_names"]),
            }

    def Predict(self, request: predict_pb2.PredictRequest, context: grpc.RpcContext):
        model_name = request.model_spec.name
        if model_name not in self.model_map:
            raise PythieServingException(
                f"Unknown model: {model_name}. This pythie-serving instance can only "
                f'serve one of the following: {",".join(self.model_map.keys())}'
            )

        model_dict = self.model_map[model_name]

        model = model_dict["model"]
        features_names = model_dict["feature_names"]
        nb_features = model_dict["nb_features"]

        samples = parse_sample(request.inputs, features_names, nb_features)

        return model.predict(DMatrix(samples)).reshape(-1)
