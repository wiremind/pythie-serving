import csv
import json
import pickle
from unittest import TestCase

import numpy as np
import treelite_runtime
import xgboost as xgb
from google.protobuf import text_format
from grpc import insecure_channel
from numpy.testing import assert_array_almost_equal

from pythie_serving.server import create_grpc_server
from pythie_serving.tensorflow_proto.tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc,
)
from pythie_serving.tensorflow_proto.tensorflow_serving.config import (
    model_server_config_pb2,
)
from pythie_serving.utils import (
    get_csv_type,
    make_ndarray_from_tensor,
    make_tensor_proto,
)

from .generate_test_resources import RESOURCES_PATH


class PredictionServiceServicerTestCaseMixin:
    model_platform = None
    server = None
    model = None
    metadata = None

    def setUp(self) -> None:

        # serve model locally
        model_server_config = model_server_config_pb2.ModelServerConfig()
        text_format.Parse(
            f"""
        model_config_list {{
            config {{
                name: "{self.model_platform}-model",
                base_path: "{RESOURCES_PATH}/{self.model_platform}-model",
                model_platform: "{self.model_platform}"
            }}
        }}
        """,
            model_server_config,
        )
        self.server = create_grpc_server(
            model_server_config=model_server_config,
            worker_count=1,
            port=9999,
            maximum_concurrent_rpcs=1,
        )
        self.server.start()

        # also deserialize the model to call directly
        with open(f"{RESOURCES_PATH}/{self.model_platform}-model/metadata.json") as f:
            self.metadata = json.loads(f.read())
        self.model = self._deserialize_model()

    def tearDown(self) -> None:
        self.server.stop(None)

    def test_serving(self):
        """Checks that the predictions returned by the deployed model are the same as the python class."""

        inputs, concatenated_inputs = self._generate_test_inputs(nb_samples=50)

        # create prediction request
        request = predict_pb2.PredictRequest()
        request.model_spec.name = f"{self.model_platform}-model"
        request.model_spec.signature_name = "serving_default"
        for feature_name, _array in inputs.items():
            request.inputs[feature_name].CopyFrom(make_tensor_proto(_array))

        # grpc call to model
        channel = insecure_channel("localhost:9999")
        response = prediction_service_pb2_grpc.PredictionServiceStub(channel).Predict(request)
        outputs = make_ndarray_from_tensor(response.outputs["predictions"])

        # call python model direclty
        expected = self._model_predict(concatenated_inputs)

        assert_array_almost_equal(outputs, expected, decimal=3)

    def _deserialize_model(self):
        with open(f"{RESOURCES_PATH}/{self.model_platform}-model/{self.model_platform}-model.pickled", "rb") as f:
            model = pickle.loads(f.read())
        return model

    def _model_predict(self, inputs):
        """Defined separately because must be overriden on some platforms."""
        return self.model.predict(inputs)

    def _generate_test_inputs(self, nb_samples):
        inputs = {
            feature_name: 100 * np.random.random((nb_samples, 1)) for feature_name in self.metadata["feature_names"]
        }
        concatenated_inputs = np.concatenate(list(inputs.values()), dtype=float, axis=1)
        return inputs, concatenated_inputs


class SklearnPredictionServiceServicerTestCase(PredictionServiceServicerTestCaseMixin, TestCase):
    model_platform = "sklearn"

    def _generate_test_inputs(self, nb_samples):
        feature_names = self.metadata["feature_names"]
        # last column is categorical
        inputs = {
            **{feature_name: 100 * np.random.random((nb_samples, 1)) for feature_name in feature_names},
            feature_names[-1]: np.array(np.random.choice(self.model["encoder"].classes_, nb_samples)).reshape(-1, 1),
        }
        concatenated_inputs = np.concatenate(list(inputs.values()), dtype=object, axis=1)

        return inputs, concatenated_inputs


class LGBMPredictionServiceServicerTestCase(PredictionServiceServicerTestCaseMixin, TestCase):
    model_platform = "lightgbm"


class XGBoostPredictionServiceServicerTestCase(PredictionServiceServicerTestCaseMixin, TestCase):
    model_platform = "xgboost"

    def _model_predict(self, inputs):
        d_matrix = xgb.DMatrix(inputs, feature_names=self.metadata["feature_names"])
        return self.model.predict(d_matrix, ntree_limit=self.model.best_ntree_limit).reshape(-1, 1)


class TreelitePredictionServiceServicerTestCase(PredictionServiceServicerTestCaseMixin, TestCase):
    model_platform = "treelite"

    def _deserialize_model(self):
        model = treelite_runtime.Predictor(
            libpath=f"{RESOURCES_PATH}/{self.model_platform}-model/{self.model_platform}-model.so",
        )
        return model

    def _model_predict(self, inputs):
        d_matrix = treelite_runtime.DMatrix(inputs)
        return self.model.predict(d_matrix).reshape(-1)


class TablePredictionServiceServicerTestCase(PredictionServiceServicerTestCaseMixin, TestCase):
    model_platform = "table"

    def _deserialize_model(self):
        with open(f"{RESOURCES_PATH}/{self.model_platform}-model/{self.model_platform}-model.csv") as csvfile:
            reader = csv.DictReader(csvfile)

            # convert data type as csv reader only returns string
            table = {}
            table_type_mapping = get_csv_type(self.metadata["data_type"])

            for row in reader:
                key = tuple(
                    table_type_mapping[feature_name](row[feature_name])
                    for feature_name in self.metadata["feature_names"]
                )
                value = table_type_mapping[self.metadata["target_name"]](row[self.metadata["target_name"]])
                table[key] = value
        return table

    def _model_predict(self, inputs):
        output = np.empty((np.shape(inputs)[0],), np.int)
        for idx, sample in enumerate(inputs):
            pred = self.model[tuple(feature_value for feature_value in sample)]

            output[idx] = pred
        return output

    def _generate_test_inputs(self, nb_samples):
        # table is a lookup table so keys can't be random
        keys = np.array(list(self.model.keys()))
        concatenated_inputs = np.array(keys)[np.random.choice(50, 50)]
        inputs = {
            feature_name: concatenated_inputs[:, i].reshape(-1, 1)
            for i, feature_name in enumerate(self.metadata["feature_names"])
        }

        return inputs, concatenated_inputs
