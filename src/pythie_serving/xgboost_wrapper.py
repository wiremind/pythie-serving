import pickle

from numpy.typing import NDArray
from xgboost import DMatrix

from pythie_serving.abstract_wrapper import (
    AbstractPythieServingPredictionServiceServicer,
    ModelSpecs,
)

from .tensorflow_proto.tensorflow_serving.config.model_server_config_pb2 import (
    ModelConfig,
)


class XGBoostPredictionServiceServicer(AbstractPythieServingPredictionServiceServicer):
    model_file_extension = ".pickled"

    def _create_model_specs(self, model_config: ModelConfig) -> ModelSpecs:
        with open(self._get_model_path(model_config), "rb") as opened_model:
            model = pickle.load(opened_model)
            model.set_param({"predictor": "cpu_predictor"})
            return {
                "model": model,
                "feature_names": model.feature_names,
                "nb_features": len(model.feature_names),
                "samples_dtype": float,
            }

    def _predict(self, model_specs: ModelSpecs, samples: NDArray) -> NDArray:
        model = model_specs["model"]
        return model.predict(
            DMatrix(samples, feature_names=model_specs["feature_names"]),
            ntree_limit=model.best_ntree_limit,
        ).reshape((-1, 1))
