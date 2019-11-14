from concurrent import futures
import os
import logging

import grpc

from .tensorflow_proto.tensorflow_serving.apis import prediction_service_pb2_grpc
from .xgboost_wrapper import XGBoostPredictionServiceServicer


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


def serve(*, models_root_path: str, model_type: str, worker_count: int, port: int, _logger: logging.Logger):
    if model_type == 'xgboost':
        servicer_cls = XGBoostPredictionServiceServicer
    else:
        raise ValueError(f'Unsupported model_type {model_type}')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=worker_count))
    server.add_insecure_port(f'[::]:{port}')

    model_paths = []
    for dirpath, _, filenames in os.walk(models_root_path):
        for filename in filenames:
            model_name, extension = os.path.splitext(filename)
            if extension == '.model':
                model_paths.append((model_name, os.path.join(dirpath, filename)))
        break

    servicer = servicer_decorator(_logger, servicer_cls(logger=_logger, model_paths=model_paths))
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(servicer, server)

    server.start()
    server.wait_for_termination()
