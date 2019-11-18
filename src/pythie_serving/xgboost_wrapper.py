import pickle
import logging

import grpc
from xgboost import DMatrix

from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .tensorflow_proto.tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from .utils import make_tensor_proto, make_ndarray_from_tensor
from .exceptions import PythieServingException


class XGBoostPredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer):

    def __init__(self, *, logger: logging.Logger, model_server_config: model_server_config_pb2.ModelServerConfig):
        self.logger = logger
        self.model_map = {}
        for model_config in model_server_config.model_config_list.config:
            with open(model_config.base_path, 'rb') as opened_model:
                model = pickle.load(opened_model)
                self.model_map[model_config.name] = {'model': model, 'feature_names': model.feature_names}

    def Predict(self, request: predict_pb2.PredictRequest, context: grpc.RpcContext):
        model_name = request.model_spec.name
        if model_name not in self.model_map:
            raise PythieServingException(f'Unknown model: {model_name}. This pythie-serving instance can only '
                                         f'serve one of the following: {",".join(self.model_map.keys())}')

        model_dict = self.model_map[model_name]

        features_names, zip_components = model_dict['feature_names'], []
        for feature_name in features_names:
            if feature_name not in request.inputs:
                raise PythieServingException(f'{feature_name} not set in the predict request')
            zip_components.append(make_ndarray_from_tensor(request.inputs[feature_name]))

        if len(set(len(z) for z in zip_components)) != 1:
            raise PythieServingException('All input vectors should have the same length')

        outputs = model_dict['model'].predict(DMatrix(list(zip(*zip_components)), feature_names=features_names))

        tf_response = predict_pb2.PredictResponse(
            model_spec=request.model_spec, outputs={'predictions': make_tensor_proto(outputs)}
        )

        return tf_response