# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from pythie_serving.tensorflow_proto.tensorflow.contrib.verbs import verbs_service_pb2 as tensorflow_dot_contrib_dot_verbs_dot_verbs__service__pb2


class VerbsServiceStub(object):
  """//////////////////////////////////////////////////////////////////////////////

  VerbsService

  //////////////////////////////////////////////////////////////////////////////

  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetRemoteAddress = channel.unary_unary(
        '/tensorflow.VerbsService/GetRemoteAddress',
        request_serializer=tensorflow_dot_contrib_dot_verbs_dot_verbs__service__pb2.GetRemoteAddressRequest.SerializeToString,
        response_deserializer=tensorflow_dot_contrib_dot_verbs_dot_verbs__service__pb2.GetRemoteAddressResponse.FromString,
        )


class VerbsServiceServicer(object):
  """//////////////////////////////////////////////////////////////////////////////

  VerbsService

  //////////////////////////////////////////////////////////////////////////////

  """

  def GetRemoteAddress(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_VerbsServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetRemoteAddress': grpc.unary_unary_rpc_method_handler(
          servicer.GetRemoteAddress,
          request_deserializer=tensorflow_dot_contrib_dot_verbs_dot_verbs__service__pb2.GetRemoteAddressRequest.FromString,
          response_serializer=tensorflow_dot_contrib_dot_verbs_dot_verbs__service__pb2.GetRemoteAddressResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorflow.VerbsService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
