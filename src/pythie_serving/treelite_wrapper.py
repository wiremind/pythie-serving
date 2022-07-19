import json
import os

from numpy.typing import NDArray
from treelite_runtime import DMatrix, Predictor

from pythie_serving.abstract_wrapper import (
    AbstractPythieServingPredictionServiceServicer,
    ModelSpecs,
)

from .tensorflow_proto.tensorflow_serving.config.model_server_config_pb2 import (
    ModelConfig,
)


class TreelitePredictionServiceServicer(AbstractPythieServingPredictionServiceServicer):
    model_file_extension = ".so"

    def _create_model_specs(self, model_config: ModelConfig) -> ModelSpecs:
        model = Predictor(
            libpath=str(self._get_model_path(model_config)),
            nthread=int(os.environ.get("TREELITE_NTHREAD", 1)),
        )

        with open(self._get_metadata_path(model_config)) as f:
            metadata = json.load(f)

        return {
            "model": model,
            "feature_names": metadata["feature_names"],
            "nb_features": len(metadata["feature_names"]),
            "samples_dtype": float,
        }

    def _predict(self, model_specs: ModelSpecs, samples: NDArray) -> NDArray:
        return model_specs["model"].predict(DMatrix(samples)).reshape(-1)
