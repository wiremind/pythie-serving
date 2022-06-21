import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict

import grpc
import numpy as np

from .exceptions import PythieServingException
from .tensorflow_proto.tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc,
)
from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .utils import get_csv_type, parse_sample


class TablePredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer):
    def __init__(
        self,
        *,
        logger: logging.Logger,
        model_server_config: model_server_config_pb2.ModelServerConfig,
    ):
        self.logger = logger
        self.table_map: Dict[str, Any] = {}
        for table_config in model_server_config.model_config_list.config:

            table_path = Path().joinpath(table_config.base_path, table_config.name + ".csv")
            if not table_path.exists():
                raise PythieServingException(f"CSV table {table_config.name} not found " f"at {table_path}")

            metadata_path = Path().joinpath(table_config.base_path, "metadata.json")
            if not metadata_path.exists():
                raise PythieServingException(f"Metadata file metadata.json not found " f"at {metadata_path}")

            with open(metadata_path) as f:
                metadata = json.load(f)

            if metadata["data_type"][metadata["target_name"]] != "int":
                raise PythieServingException(
                    f"Can only serve integer target, but "
                    f"{metadata['data_type'][metadata['target_name']]} was specified."
                )

            with open(table_path) as csvfile:
                reader = csv.DictReader(csvfile)

                # convert data type as csv reader only returns string
                table = {}
                table_type_mapping = get_csv_type(metadata["data_type"])

                for row in reader:
                    key = tuple(
                        table_type_mapping[feature_name](row[feature_name])
                        for feature_name in metadata["feature_names"]
                    )
                    value = table_type_mapping[metadata["target_name"]](row[metadata["target_name"]])
                    table[key] = value

            self.table_map[table_config.name] = {
                "table": table,
                "feature_names": metadata["feature_names"],
                "nb_features": len(metadata["feature_names"]),
            }

    def Predict(self, request: predict_pb2.PredictRequest, context: grpc.RpcContext):
        table_name = request.model_spec.name
        if table_name not in self.table_map:
            raise PythieServingException(
                f"Unknown table: {table_name}. This pythie-serving instance can only "
                f'serve one of the following: {",".join(self.table_map.keys())}'
            )

        table_dict = self.table_map[table_name]
        table = table_dict["table"]
        features_names = table_dict["feature_names"]
        nb_features = table_dict["nb_features"]

        samples = parse_sample(request.inputs, features_names, nb_features, object)

        output = np.empty((np.shape(samples)[0],), np.int)
        for idx, sample in enumerate(samples):
            try:
                pred = table[tuple(feature_value for feature_value in sample)]
            except KeyError:
                raise PythieServingException(
                    f"No prediction found in table {table_name} for given features: " f"{features_names} = {sample}."
                )

            output[idx] = pred

        return output
