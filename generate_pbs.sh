#!/usr/bin/env bash

set -e

current_path="$PWD"

TF_URL="https://github.com/tensorflow/tensorflow.git"
TF_SERVING_URL="https://github.com/tensorflow/serving.git"
TF_COMMIT=$1

# Retrieve tensorflow and tensorflow_serving repos
mkdir -p /tmp/scratch
cd /tmp/scratch
git clone --branch "v$TF_COMMIT" "$TF_URL"
git clone --branch "$TF_COMMIT" "$TF_SERVING_URL"
cd ..

# Reorganize contents to make it easy to compile protos.
# This way the final results will have correct relative paths
# to each other.
mkdir -p workspace
mv /tmp/scratch/tensorflow/tensorflow /tmp/workspace/
mv /tmp/scratch/serving/tensorflow_serving /tmp/workspace/

# Install prerequisite packages
. ~/.virtualenvs/grpc-build/bin/activate
pip install --upgrade pip protobuf grpcio grpcio-tools mypy-protobuf

# Create the python package of pb2 interfaces
mkdir -p /tmp/tensorflow_proto
cd /tmp/workspace
## Compile protobufs
find . -name '*.proto' -exec \
  python -m grpc_tools.protoc -I./ \
    --python_out=../tensorflow_proto/ \
    --grpc_python_out=../tensorflow_proto/ \
    --mypy_out=../tensorflow_proto/ \
    {} ';'
find /tmp/tensorflow_proto/ -type d -exec touch {}/__init__.py ';'
find /tmp/tensorflow_proto/ -name '*.py' -exec sed -i -- 's/from tensorflow\./from pythie_serving.tensorflow_proto.tensorflow./g' {} ';'
find /tmp/tensorflow_proto/ -name '*.py' -exec sed -i -- 's/from tensorflow_serving\./from pythie_serving.tensorflow_proto.tensorflow_serving./g' {} ';'

mv /tmp/tensorflow_proto/ "$current_path"

rm -rf /tmp/workspace
rm -rf /tmp/scratch
rm -rf /tmp/tensorflow_proto
