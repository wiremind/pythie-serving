from concurrent import futures
import logging

import grpc

from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .tensorflow_proto.tensorflow_serving.apis import prediction_service_pb2_grpc
from .xgboost_wrapper import XGBoostPredictionServiceServicer
from .exceptions import PythieServingException


def servicer_decorator(_logger, servicer):
    for fn_name in ['Predict']:
        def fn_decorator(fn):
            def wrapper(request, context):
                _logger.info(f'Serving model {request.model_spec.name}/{request.model_spec.signature_name}')
                try:
                    return fn(request, context)
                except Exception as e:
                    _logger.error(f'Failed to serve {request} because: {e}')
                    raise
            return wrapper

        setattr(servicer, fn_name, fn_decorator(getattr(servicer, fn_name)))
    return servicer


def serve(*, model_server_config: model_server_config_pb2.ModelServerConfig, worker_count: int,
          port: int, _logger: logging.Logger):
    model_platforms = set(c.model_platform for c in model_server_config.model_config_list.config)
    if len(model_platforms) > 1:
        raise PythieServingException('Only one model_plateform can be served at a time')

    model_platform = model_platforms.pop()
    if model_platform == 'xgboost':
        servicer_cls = XGBoostPredictionServiceServicer
    else:
        raise ValueError(f'Unsupported model_platform {model_platform}')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=worker_count))
    server.add_insecure_port(f'[::]:{port}')

    servicer = servicer_decorator(_logger, servicer_cls(logger=_logger, model_server_config=model_server_config))
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(servicer, server)

    server.start()
    server.wait_for_termination()
