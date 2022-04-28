from concurrent import futures
import logging
import time
from typing import Optional

import grpc

from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2
from .tensorflow_proto.tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from .utils import make_tensor_proto
from .exceptions import PythieServingException


def servicer_decorator(_logger, servicer):
    for fn_name in ['Predict']:
        def fn_decorator(fn):
            def wrapper(request, context):
                start = time.time()
                try:
                    pred = fn(request, context)
                    predict_response = predict_pb2.PredictResponse(
                        model_spec=request.model_spec, outputs={'predictions': make_tensor_proto(pred)}
                    )
                except Exception as e:
                    _logger.error(f'Failed to serve because: {e}')
                    raise
                else:
                    duration = time.time() - start
                    _logger.info(
                        f'Served model {request.model_spec.name}/{request.model_spec.signature_name}: '
                        f'{len(pred)} predictions in {duration:.2f} seconds ({len(pred) / duration:.2f} pred/sec) '
                    )
                    return predict_response
            return wrapper

        setattr(servicer, fn_name, fn_decorator(getattr(servicer, fn_name)))
    return servicer


def serve(*, model_server_config: model_server_config_pb2.ModelServerConfig, worker_count: int,
          port: int, maximum_concurrent_rpcs: Optional[int], _logger: logging.Logger):
    model_platforms = set(c.model_platform for c in model_server_config.model_config_list.config)
    if len(model_platforms) > 1:
        raise PythieServingException('Only one model_plateform can be served at a time')

    model_platform = model_platforms.pop()
    # import in code to avoid loading too many python libraries in memory
    if model_platform == 'xgboost':
        if worker_count > 1:
            raise ValueError(f'Model platform {model_platform} is not thread safe')
        from .xgboost_wrapper import XGBoostPredictionServiceServicer
        servicer_cls = XGBoostPredictionServiceServicer
    elif model_platform == 'lightgbm':
        from .lightgbm_wrapper import LightGBMPredictionServiceServicer
        servicer_cls = LightGBMPredictionServiceServicer
    elif model_platform == 'treelite':
        from .treelite_wrapper import TreelitePredictionServiceServicer
        servicer_cls = TreelitePredictionServiceServicer
    elif model_platform == 'sklearn':
        from .sklearn_wrapper import SklearnPredictionServiceServicer
        servicer_cls = SklearnPredictionServiceServicer
    else:
        raise ValueError(f'Unsupported model platform {model_platform}')

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=worker_count), maximum_concurrent_rpcs=maximum_concurrent_rpcs
    )
    server.add_insecure_port(f'[::]:{port}')

    servicer = servicer_decorator(_logger, servicer_cls(logger=_logger, model_server_config=model_server_config))
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(servicer, server)

    server.start()
    server.wait_for_termination()
