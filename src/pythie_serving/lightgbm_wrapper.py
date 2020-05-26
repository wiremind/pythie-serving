import pickle
import logging

import grpc

from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .tensorflow_proto.tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from .utils import make_ndarray_from_tensor
from .exceptions import PythieServingException


class LgbEnsemble:
    """
    Average LightGBM models
    """
    def __init__(self, lgb_models):
        """
        :param lgb_models: list of LightGBM boosters
        """
        self.sub_models = lgb_models
        self.feature_names = lgb_models[0].feature_name()

    def predict(self, X):
        reg = self.sub_models[0]
        preds = reg.predict(X, num_iteration=reg.best_iteration)
        for reg in self.sub_models[1:]:
            preds += reg.predict(X, num_iteration=reg.best_iteration)
        preds /= len(self.sub_models)
        return preds


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'LgbEnsemble':
            return LgbEnsemble
        return super().find_class(module, name)


class LightGBMPredictionServiceServicer(prediction_service_pb2_grpc.PredictionServiceServicer):

    def __init__(self, *, logger: logging.Logger, model_server_config: model_server_config_pb2.ModelServerConfig):
        self.logger = logger
        self.model_map = {}
        for model_config in model_server_config.model_config_list.config:
            with open(model_config.base_path, 'rb') as opened_model:
                model = CustomUnpickler(opened_model).load()
                self.model_map[model_config.name] = {'model': model, 'feature_names': model.feature_names}

    def Predict(self, request: predict_pb2.PredictRequest, context: grpc.RpcContext):
        model_name = request.model_spec.name
        if model_name not in self.model_map:
            raise PythieServingException(f'Unknown model: {model_name}. This pythie-serving instance can only '
                                         f'serve one of the following: {",".join(self.model_map.keys())}')

        model_dict = self.model_map[model_name]

        features_names = model_dict['feature_names']
        samples = None
        for feature_name in features_names:
            if feature_name not in request.inputs:
                raise PythieServingException(f'{feature_name} not set in the predict request')

            nd_array = make_ndarray_from_tensor(request.inputs[feature_name])
            if len(nd_array.shape) != 2 or nd_array.shape[1] != 1:
                raise PythieServingException('All input vectors should be 1D tensor')

            if samples is None:
                samples = [[] for _ in range(nd_array.shape[0])]

            for sample_index, value in enumerate(nd_array):
                samples[sample_index].append(value[0])

        model = model_dict['model']
        return model.predict(samples)
