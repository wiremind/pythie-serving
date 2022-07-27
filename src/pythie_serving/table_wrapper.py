import csv
import json

import numpy as np
from numpy.typing import NDArray

from pythie_serving.abstract_wrapper import (
    AbstractPythieServingPredictionServiceServicer,
    ModelSpecs,
)

from .exceptions import PythieServingException
from .tensorflow_proto.tensorflow_serving.config.model_server_config_pb2 import (
    ModelConfig,
)
from .utils import get_csv_type


class TablePredictionServiceServicer(AbstractPythieServingPredictionServiceServicer):
    model_file_extension = ".csv"

    def _create_model_specs(self, model_config: ModelConfig) -> ModelSpecs:

        with open(self._get_metadata_path(model_config)) as f:
            metadata = json.load(f)

        if metadata["data_type"][metadata["target_name"]] != "int":
            raise PythieServingException(
                f"Can only serve integer target, but "
                f"{metadata['data_type'][metadata['target_name']]} was specified."
            )

        with open(self._get_model_path(model_config)) as csvfile:
            reader = csv.DictReader(csvfile)

            # convert data type as csv reader only returns string
            table = {}
            table_type_mapping = get_csv_type(metadata["data_type"])

            for row in reader:
                key = tuple(
                    table_type_mapping[feature_name](row[feature_name]) for feature_name in metadata["feature_names"]
                )
                value = table_type_mapping[metadata["target_name"]](row[metadata["target_name"]])
                table[key] = value

        return {
            "model": table,
            "feature_names": metadata["feature_names"],
            "nb_features": len(metadata["feature_names"]),
            "samples_dtype": object,
        }

    def _predict(self, model_specs: ModelSpecs, samples: NDArray) -> NDArray:

        output = np.empty((np.shape(samples)[0],), dtype=int)
        for idx, sample in enumerate(samples):
            try:
                pred = model_specs["model"][tuple(feature_value for feature_value in sample)]
            except KeyError:
                raise PythieServingException(
                    f"No prediction found in table for given features: " f"{model_specs['feature_names']} = {sample}."
                )

            output[idx] = pred

        return output
