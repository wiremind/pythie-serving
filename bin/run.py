import sys
import logging
import os
from argparse import ArgumentParser
from logging.config import dictConfig

from pythie_serving import serve

if __name__ == '__main__':
    model_choice_set = {'xgboost'}
    model_choice_str = ','.join(model_choice_set)

    parser = ArgumentParser(description=f'A GRPC server to serve different kind of model amongst: {model_choice_str}')
    parser.add_argument('--model-type', type=str, default='xgboost')
    parser.add_argument('--models-root-path', type=str, default='./models/',
                        help='The root path where to find the models. The server will load every file with '
                             '.model extension it founds inside this directory')
    parser.add_argument('--worker-count', default=10, type=int, help='Number of concurrent threads for the GRPC server')
    parser.add_argument('--port', default=9090, type=int, help='Port number to listen to')

    dictConfig({
        'disable_existing_loggers': True,
        'version': 1,
        'formatters': {
            'console_formatter': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
                '()': 'logging.Formatter'
            }
        },
        'handlers': {
            'stdout_handler': {
                'class': 'logging.StreamHandler',
                'formatter': 'console_formatter',
                'level': 'INFO',
                'stream': sys.stdout
            }
        },
        'root': {'level': 'INFO', 'handlers': ['stdout_handler'], 'formatter': ['console_formatter']}
    })
    logger = logging.getLogger('pythie_serving')

    ns = parser.parse_args()
    if ns.model_type not in model_choice_set:
        raise ValueError(f'Unsupported model_type {ns.model_type}, choose amongst: {model_choice_str}')

    if not os.path.exists(ns.models_root_path):
        raise ValueError(f'Model not found at {ns.models_root_path}')

    serve(models_root_path=ns.models_root_path, model_type=ns.model_type, worker_count=ns.worker_count, port=ns.port,
          _logger=logger)
