from concurrent import futures
from typing import Optional, Type

import grpc

from pythie_serving.abstract_wrapper import (
    AbstractPythieServingPredictionServiceServicer,
)

from .exceptions import PythieServingException
from .tensorflow_proto.tensorflow_serving.apis import prediction_service_pb2_grpc
from .tensorflow_proto.tensorflow_serving.config import model_server_config_pb2


def create_grpc_server(
    *,
    model_server_config: model_server_config_pb2.ModelServerConfig,
    worker_count: int,
    port: int,
    maximum_concurrent_rpcs: Optional[int],
) -> grpc.server:
    model_platforms = {c.model_platform for c in model_server_config.model_config_list.config}
    if len(model_platforms) > 1:
        raise PythieServingException("Only one model_plateform can be served at a time")

    model_platform = model_platforms.pop()
    # import in code to avoid loading too many python libraries in memory
    servicer_cls: Type[AbstractPythieServingPredictionServiceServicer]
    if model_platform == "xgboost":
        if worker_count > 1:
            raise ValueError(f"Model platform {model_platform} is not thread safe")
        from .xgboost_wrapper import XGBoostPredictionServiceServicer

        servicer_cls = XGBoostPredictionServiceServicer
    elif model_platform == "lightgbm":
        from .lightgbm_wrapper import LightGBMPredictionServiceServicer

        servicer_cls = LightGBMPredictionServiceServicer
    elif model_platform == "treelite":
        from .treelite_wrapper import TreelitePredictionServiceServicer

        servicer_cls = TreelitePredictionServiceServicer
    elif model_platform == "sklearn":
        from .sklearn_wrapper import SklearnPredictionServiceServicer

        servicer_cls = SklearnPredictionServiceServicer
    elif model_platform == "table":
        from .table_wrapper import TablePredictionServiceServicer

        servicer_cls = TablePredictionServiceServicer
    else:
        raise ValueError(f"Unsupported model platform {model_platform}")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=worker_count),
        maximum_concurrent_rpcs=maximum_concurrent_rpcs,
    )
    server.add_insecure_port(f"[::]:{port}")

    servicer = servicer_cls(model_server_config=model_server_config)
    prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(servicer, server)

    return server
