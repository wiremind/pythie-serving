import os
import pickle

from lightgbm import Booster
from numpy.typing import NDArray

from pythie_serving.abstract_wrapper import (
    AbstractPythieServingPredictionServiceServicer,
    ModelSpecs,
)

from .tensorflow_proto.tensorflow_serving.config.model_server_config_pb2 import (
    ModelConfig,
    ModelServerConfig,
)


class LightGBMPredictionServiceServicer(AbstractPythieServingPredictionServiceServicer):
    model_file_extension = ".pickled"

    def __init__(self, *, model_server_config: ModelServerConfig):
        super().__init__(model_server_config=model_server_config)
        self.nthread = int(os.environ.get("LGBM_NTHREAD", 0))

    def _create_model_specs(self, model_config: ModelConfig) -> ModelSpecs:
        with open(self._get_model_path(model_config), "rb") as opened_model:
            model = pickle.load(opened_model)

            if isinstance(model, Booster):
                feature_names = model.feature_name()
            else:
                feature_names = model.feature_names

            return {
                "model": model,
                "feature_names": feature_names,
                "nb_features": len(feature_names),
                "samples_dtype": float,
            }

    def _predict(self, model_specs: ModelSpecs, samples: NDArray) -> NDArray:
        return model_specs["model"].predict(samples, nthread=self.nthread)
