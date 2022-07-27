# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from pythie_serving.tensorflow_proto.tensorflow.compiler.xla.pjrt.distributed import protocol_pb2 as tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2


class DistributedRuntimeServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Connect = channel.unary_unary(
                '/xla.DistributedRuntimeService/Connect',
                request_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ConnectRequest.SerializeToString,
                response_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ConnectResponse.FromString,
                )
        self.EnumerateDevices = channel.unary_unary(
                '/xla.DistributedRuntimeService/EnumerateDevices',
                request_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.EnumerateDevicesRequest.SerializeToString,
                response_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.EnumerateDevicesResponse.FromString,
                )
        self.Heartbeat = channel.unary_unary(
                '/xla.DistributedRuntimeService/Heartbeat',
                request_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.HeartbeatRequest.SerializeToString,
                response_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.HeartbeatResponse.FromString,
                )
        self.Shutdown = channel.unary_unary(
                '/xla.DistributedRuntimeService/Shutdown',
                request_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ShutdownRequest.SerializeToString,
                response_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ShutdownResponse.FromString,
                )
        self.KeyValueGet = channel.unary_unary(
                '/xla.DistributedRuntimeService/KeyValueGet',
                request_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueGetRequest.SerializeToString,
                response_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueGetResponse.FromString,
                )
        self.KeyValueSet = channel.unary_unary(
                '/xla.DistributedRuntimeService/KeyValueSet',
                request_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueSetRequest.SerializeToString,
                response_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueSetResponse.FromString,
                )


class DistributedRuntimeServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Connect(self, request, context):
        """Connects a node to the distributed coordinator node. Blocks until all tasks
        have connected. The service receives the number of nodes to expect as an
        option passed to its constructor.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def EnumerateDevices(self, request, context):
        """Blocking enumeration of devices, used by the GPU backend only.
        In parallel, all clients call EnumerateDevices() with their local device
        topology, and receive back a global topology in response.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Heartbeat(self, request, context):
        """Health-checking RPC. Workers send heartbeats to the coordinator at regular
        intervals. If the worker does not hear from the coordinator or the
        coordinator does not hear from the tasks, the tasks abort.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Shutdown(self, request, context):
        """Shutdown RPC. Workers send this RPC when they are ready to shut down; the
        RPC blocks until all tasks have indicated they are ready to shut down,
        or a timeout is reached.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def KeyValueGet(self, request, context):
        """Simple key-value store used for sharing configuration data.
        For example, when using NCCL to communicate between multiple GPUs,
        the NCCL communicator IDs are stored here.

        Looks up a key in the key-value service. Blocks until the key is present
        or until `timeout` expires.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def KeyValueSet(self, request, context):
        """Updates the value associated with a key.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DistributedRuntimeServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Connect': grpc.unary_unary_rpc_method_handler(
                    servicer.Connect,
                    request_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ConnectRequest.FromString,
                    response_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ConnectResponse.SerializeToString,
            ),
            'EnumerateDevices': grpc.unary_unary_rpc_method_handler(
                    servicer.EnumerateDevices,
                    request_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.EnumerateDevicesRequest.FromString,
                    response_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.EnumerateDevicesResponse.SerializeToString,
            ),
            'Heartbeat': grpc.unary_unary_rpc_method_handler(
                    servicer.Heartbeat,
                    request_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.HeartbeatRequest.FromString,
                    response_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.HeartbeatResponse.SerializeToString,
            ),
            'Shutdown': grpc.unary_unary_rpc_method_handler(
                    servicer.Shutdown,
                    request_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ShutdownRequest.FromString,
                    response_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ShutdownResponse.SerializeToString,
            ),
            'KeyValueGet': grpc.unary_unary_rpc_method_handler(
                    servicer.KeyValueGet,
                    request_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueGetRequest.FromString,
                    response_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueGetResponse.SerializeToString,
            ),
            'KeyValueSet': grpc.unary_unary_rpc_method_handler(
                    servicer.KeyValueSet,
                    request_deserializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueSetRequest.FromString,
                    response_serializer=tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueSetResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'xla.DistributedRuntimeService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DistributedRuntimeService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Connect(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/xla.DistributedRuntimeService/Connect',
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ConnectRequest.SerializeToString,
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ConnectResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def EnumerateDevices(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/xla.DistributedRuntimeService/EnumerateDevices',
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.EnumerateDevicesRequest.SerializeToString,
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.EnumerateDevicesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Heartbeat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/xla.DistributedRuntimeService/Heartbeat',
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.HeartbeatRequest.SerializeToString,
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.HeartbeatResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Shutdown(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/xla.DistributedRuntimeService/Shutdown',
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ShutdownRequest.SerializeToString,
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.ShutdownResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KeyValueGet(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/xla.DistributedRuntimeService/KeyValueGet',
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueGetRequest.SerializeToString,
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueGetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def KeyValueSet(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/xla.DistributedRuntimeService/KeyValueSet',
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueSetRequest.SerializeToString,
            tensorflow_dot_compiler_dot_xla_dot_pjrt_dot_distributed_dot_protocol__pb2.KeyValueSetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
