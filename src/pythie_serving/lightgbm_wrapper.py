import logging
import os
import pickle
from typing import List, Optional, Any

import grpc
from lightgbm import Booster

from .exceptions import PythieServingException
from .tensorflow_proto.tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc,
)
from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .utils import make_ndarray_from_tensor


class LightGBMPredictionServiceServicer(
    prediction_service_pb2_grpc.PredictionServiceServicer
):
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
                pandas_categorical = model.pandas_categorical

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
                    "pandas_categorical": pandas_categorical,
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
        pandas_categorical = model_dict["pandas_categorical"]
        features_names = model_dict["feature_names"]
        pd_categorical_features: List[str] = []
        samples: Optional[List[List[Any]]] = None
        for feature_name in features_names:
            if feature_name not in request.inputs:
                raise PythieServingException(
                    f"{feature_name} not set in the predict request"
                )

            nd_array = make_ndarray_from_tensor(request.inputs[feature_name])
            # get features categories from model if feature dtype is object ("string")
            feature_categories = {}
            if nd_array.dtype == object:
                if pandas_categorical is None:
                    raise PythieServingException(
                        f"{feature_name} feature has type 'object' but "
                        f"there is no saved pandas categories from model"
                    )
                # lgbm save categories in the same order categorical features appear in model.feature_name()
                feature_categories = {
                    category_name: category_position
                    for category_position, category_name in enumerate(
                        pandas_categorical[len(pd_categorical_features)]
                    )
                }
                pd_categorical_features.append(feature_name)

            if len(nd_array.shape) != 2 or nd_array.shape[1] != 1:
                raise PythieServingException("All input vectors should be 1D tensor")

            samples = samples or [[] for _ in range(nd_array.shape[0])]
            if len(feature_categories) > 0:  # get category position from its value
                for sample_index, value in enumerate(nd_array):
                    samples[sample_index].append(
                        feature_categories[value[0].decode("utf-8")]
                    )
            else:
                for sample_index, value in enumerate(nd_array):
                    samples[sample_index].append(value[0])

        kwargs = {}
        if model_dict["best_iteration"]:
            kwargs["best_iteration"] = model_dict["best_iteration"]
        return model.predict(samples, **kwargs)
