import abc
import logging
import time
from pathlib import Path
from typing import Any, ClassVar, List, TypedDict

import grpc
import numpy as np
from numpy.typing import NDArray

from .exceptions import PythieServingException
from .tensorflow_proto.tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc,
)
from .tensorflow_proto.tensorflow_serving.config.model_server_config_pb2 import (
    ModelConfig,
    ModelServerConfig,
)
from .utils import make_ndarray_from_tensor, make_tensor_proto


class ModelSpecs(TypedDict):
    model: Any
    feature_names: List[str]
    nb_features: int
    samples_dtype: Any


class AbstractPythieServingPredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer, abc.ABC):
    model_file_extension: ClassVar[str]

    def __init__(self, *, model_server_config: ModelServerConfig):
        self.logger = logging.getLogger("pythie_serving")

        self.model_map = {}
        for model_config in model_server_config.model_config_list.config:
            if not self._get_model_path(model_config).exists():
                raise PythieServingException(f"Model file for {model_config.name} not found.")
            if not self._get_metadata_path(model_config).exists():
                raise PythieServingException("Metadata file metadata.json not found ")

            self.model_map[model_config.name] = self._create_model_specs(model_config)

    def _get_model_path(self, model_config: ModelConfig) -> Path:
        return Path().joinpath(model_config.base_path, model_config.name).with_suffix(self.model_file_extension)

    @staticmethod
    def _get_metadata_path(model_config: ModelConfig) -> Path:
        return Path().joinpath(model_config.base_path, "metadata.json")

    @abc.abstractmethod
    def _create_model_specs(self, model_config: ModelConfig) -> ModelSpecs:
        """Creates the ModelSpecs for a given ModelConfig."""
        ...

    @abc.abstractmethod
    def _predict(self, model_specs: ModelSpecs, samples: NDArray) -> NDArray:
        """How to make a prediction on samples using the model object."""
        ...

    def Predict(self, request: predict_pb2.PredictRequest, context: grpc.RpcContext) -> predict_pb2.PredictResponse:
        start = time.time()
        try:
            model_name = request.model_spec.name
            if model_name not in self.model_map:
                raise PythieServingException(
                    f"Unknown model: {model_name}. This pythie-serving instance can only "
                    f'serve one of the following: {",".join(self.model_map.keys())}'
                )
            model_specs = self.model_map[model_name]
            pred = self._predict(model_specs, samples=self._parse_samples(request, model_specs))
            predict_response = predict_pb2.PredictResponse(
                model_spec=request.model_spec, outputs={"predictions": make_tensor_proto(pred)}
            )
        except Exception as e:
            self.logger.error(f"Failed to serve because: {e}")
            raise
        else:
            duration = time.time() - start
            self.logger.info(
                f"Served model {request.model_spec.name}/{request.model_spec.signature_name}: "
                f"{len(pred)} predictions in {duration:.2f} seconds ({len(pred) / duration:.2f} pred/sec) "
            )
            return predict_response

    @staticmethod
    def _parse_samples(request: predict_pb2.PredictRequest, model_specs: ModelSpecs) -> NDArray:
        nb_features = model_specs["nb_features"]
        request_inputs = request.inputs

        features_names = model_specs["feature_names"]
        if set(request_inputs) != set(features_names):
            raise PythieServingException(
                f"Features names mismatch. Expected {features_names}, got {set(request_inputs)}."
            )

        nb_samples = request_inputs[features_names[0]].tensor_shape.dim[0].size
        samples = np.empty((nb_samples, nb_features), model_specs["samples_dtype"])

        for i, feature_name in enumerate(features_names):

            if request_inputs[feature_name].tensor_shape.dim[0].size != nb_samples:
                raise PythieServingException(f"{feature_name} has invalid length.")

            nd_array = make_ndarray_from_tensor(request_inputs[feature_name])

            if len(nd_array.shape) != 2 or nd_array.shape[1] != 1:
                raise PythieServingException("All input vectors should be 1D tensor")

            samples[:, i] = nd_array.reshape(-1)

        return samples
