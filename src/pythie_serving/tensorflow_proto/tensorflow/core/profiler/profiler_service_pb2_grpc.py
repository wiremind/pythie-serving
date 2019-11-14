# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from pythie_serving.tensorflow_proto.tensorflow.core.profiler import profiler_service_pb2 as tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2


class ProfilerServiceStub(object):
  """The ProfilerService service retrieves performance information about
  the programs running on connected devices over a period of time.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Profile = channel.unary_unary(
        '/tensorflow.ProfilerService/Profile',
        request_serializer=tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2.ProfileRequest.SerializeToString,
        response_deserializer=tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2.ProfileResponse.FromString,
        )
    self.Monitor = channel.unary_unary(
        '/tensorflow.ProfilerService/Monitor',
        request_serializer=tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2.MonitorRequest.SerializeToString,
        response_deserializer=tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2.MonitorResponse.FromString,
        )


class ProfilerServiceServicer(object):
  """The ProfilerService service retrieves performance information about
  the programs running on connected devices over a period of time.
  """

  def Profile(self, request, context):
    """Starts a profiling session, blocks until it completes, and returns data.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Monitor(self, request, context):
    """Collects profiling data and returns user-friendly metrics.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ProfilerServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Profile': grpc.unary_unary_rpc_method_handler(
          servicer.Profile,
          request_deserializer=tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2.ProfileRequest.FromString,
          response_serializer=tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2.ProfileResponse.SerializeToString,
      ),
      'Monitor': grpc.unary_unary_rpc_method_handler(
          servicer.Monitor,
          request_deserializer=tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2.MonitorRequest.FromString,
          response_serializer=tensorflow_dot_core_dot_profiler_dot_profiler__service__pb2.MonitorResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorflow.ProfilerService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
