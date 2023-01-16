import json
import os
import pickle

import numpy as np
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
                "extra_specs": None,
            }

    def _predict(self, model_specs: ModelSpecs, samples: NDArray) -> NDArray:
        return model_specs["model"].predict(samples, nthread=self.nthread)


class LGBMCountServiceServicer(AbstractPythieServingPredictionServiceServicer):
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

        with open(self._get_metadata_path(model_config)) as f:
            metadata = json.load(f)
            tree_id_leaf_id_count_map = self._parse_leaf_index(metadata["leaf_index_to_count"])

        return {
            "model": model,
            "feature_names": feature_names,
            "nb_features": len(feature_names),
            "samples_dtype": float,
            "extra_specs": tree_id_leaf_id_count_map,
        }

    @staticmethod
    def _parse_leaf_index(leaf_index_to_count: dict) -> dict:
        """
        Create mapping between the leaf index outputted by lgbm.Booster to its count
        :param leaf_index_to_count: dict mapping lgbm leaf index name to its count
        :return: dict mapping a leaf's position in tree to its count for each tree
        """
        tree_id_leaf_id_count_map: dict[int, dict[int, int]] = {}
        for leaf_index, leaf_count in leaf_index_to_count.items():
            tree_id, leaf_tree_id = leaf_index.split("-")
            tree_id_leaf_id_count_map.setdefault(int(tree_id), {})
            tree_id_leaf_id_count_map[int(tree_id)][int(leaf_tree_id[1:])] = leaf_count

        return tree_id_leaf_id_count_map

    def _predict(self, model_specs: ModelSpecs, samples: NDArray) -> NDArray:
        tree_id_leaf_id_count_map: dict[int, dict[int, int]] = model_specs["extra_specs"]

        leaf_ids_in_trees: NDArray = model_specs["model"].predict(samples, nthread=self.nthread, pred_leaf=True)
        nb_trees = leaf_ids_in_trees.shape[1]

        return np.mean(
            [
                np.vectorize(tree_id_leaf_id_count_map[tree_id].get)(leaf_ids_in_trees[:, tree_id : tree_id + 1])
                for tree_id in range(nb_trees)
            ],
            axis=0,
        ).reshape(-1)
