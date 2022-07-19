import json

import cloudpickle
from numpy.typing import NDArray

from pythie_serving.abstract_wrapper import (
    AbstractPythieServingPredictionServiceServicer,
    ModelSpecs,
)

from .tensorflow_proto.tensorflow_serving.config.model_server_config_pb2 import (
    ModelConfig,
)


class SklearnPredictionServiceServicer(AbstractPythieServingPredictionServiceServicer):
    model_file_extension = ".pickled"

    def _create_model_specs(self, model_config: ModelConfig) -> ModelSpecs:
        with open(self._get_model_path(model_config), "rb") as opened_model:
            # cloudpickle used to be able to load model + modules which are not importable
            model = cloudpickle.load(opened_model)

        with open(self._get_metadata_path(model_config)) as f:
            metadata = json.load(f)

        return {
            "model": model,
            "feature_names": metadata["feature_names"],
            "nb_features": len(metadata["feature_names"]),
            "samples_dtype": object,
        }

    def _predict(self, model_specs: ModelSpecs, samples: NDArray) -> NDArray:
        return model_specs["model"].predict(samples)
