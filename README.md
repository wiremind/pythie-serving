## Goal of this library
Tensorflow's official client implementations depend on the tensorflow package as of this writing.
For instance, if you want to run their example MNIST client, you'd need to have tensorflow installed even just
to construct the prediction request predict_pb2.PredictRequest, and to call
their tf.contrib.util.make_tensor_proto function. So you can't easily use their work to make a lightweight client API
that is totally free of tensorflow.
Tensorflow being a huge library, the estimated RAM overhead of having tensorflow installed is around 150MB which makes
the client heavy.
Another goal of this project is to expose other models then tensorflow with a GRPC API following the one defined for tensorflow serving (https://github.com/tensorflow/serving).
For instance, it allows to serve a XGBoost model with the same API as the one used to request a tensorflow serving server

## How to update this lib with a new version of tensorflow_serving
We need to generate the GRPC client code ourselves from the .proto definition in
https://github.com/tensorflow/serving/tree/master/tensorflow_serving/apis

To do this,

run "sh generate_pbs.sh X.Y.Z" (you should have an already created pew virtualenv using python3.8 called 'grpc-build'),
this will generate a working python package in ./tensorflow_proto from the tag version X.Y.Z of tensorflow repo
you can then copy/paste this package inside src/pythie-serving/ and import it as you need.

This will generate the necessary stubs for type-checking

## Available model architectures
The following models can be served by pythie-serving:
* LightGBM: https://lightgbm.readthedocs.io/en/latest/
* XGBoost: https://xgboost.readthedocs.io/en/stable/
* scikit-learn: https://scikit-learn.org/stable/
* treelite: https://treelite.readthedocs.io/en/latest/
* csv table

## How to run

### Python

1. Define a `models.config` file (see: https://www.tensorflow.org/tfx/serving/serving_config):
   ```
    model_config_list {
        config {
            name: <model_name>,
            base_path: <my_base_path>,
            model_platform: <model_platform>
        }
    }
    ```
2. Run `python pythie-serving/src/pythie_serving/run.py <model_config_file_path>`.\
   The following options are available:
   * `--worker-count`: Number of concurrent threads for the GRPC server.
   * `--max-concurrent-rpcs`: The maximum number of concurrent RPCs this server.
   * `--port`: Port number to listen to.

#### Environment variables
For a treelite served model:
* `TREELITE_NTHREAD`: Number of threads to use to compute predictions
* `TREELINTE_BIND_THREADS`: Set to `0` to deactivate thread pinning. See https://treelite.readthedocs.io/en/latest/treelite-runtime-api.html
For an LGBM served model:
* `LGBM_NTHREAD`: Number of threads to use to compute predictions

### Docker

The project is published on GitHub Container Registry: https://github.com/wiremind/pythie-serving/pkgs/container/pythie-serving

## Development

### Add a new architecture type
To add a new architecture type, you need to implement a `prediction_service_pb2_grpc.PredictionServiceServicer`.
To facilitate this, pythie-serving implements an abstract `AbstractPythieServingPredictionServiceServicer` that already implements the necessary `Predict` 
method, taking a `PredictRequest` as input and outputing a `PredictResponse`.
This class leaves 2 abstract methods to be implemented:
* `_create_model_specs` to read a `ModelConfig` from the list and instantiate the model and necessary variables
* `_predict` to make a prediction using the model (as python class) on a numpy array