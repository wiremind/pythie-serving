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

run "sh generate_pbs.sh X.Y.Z" (you should have an already created pew virtualenv using python3.7 called 'grpc-build'),
this will generate a working python package in ./tensorflow_proto from the tag version X.Y.Z of tensorflow repo
you can then copy/paste this package inside src/pythie-serving/ and import it as you need

## How to run

### Python

TBD

### Docker

This repository is linked with Docker Hub with auto-build at https://hub.docker.com/r/wiremind/pythie-serving.
