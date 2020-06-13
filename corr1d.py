import tensorflow as tf
import os

_correlation1d_ops = tf.load_op_library(tf.resource_loader.get_path_to_datafile('./corr.so'))

def correlation1d(input_a, input_b, kernel_size, max_displacement, stride1, stride2, padding):
    return _correlation1d_ops.correlation1d(input_a,
                                            input_b,
                                            kernel_size,
                                            max_displacement,
                                            stride1,
                                            stride2,
                                            padding)
correlation1d = _correlation1d_ops.correlation1d

@tf.RegisterGradient('Correlation1d')
def _correlation_grad(corr1d_op, gradients):
    kernel_size = corr1d_op.get_attr('kernel_size')
    max_displacement = corr1d_op.get_attr('max_displacement')
    stride_1 = corr1d_op.get_attr('stride_1')
    stride_2 = corr1d_op.get_attr('stride_2')
    pad = corr1d_op.get_attr('pad')

    corr1d_grads = _correlation1d_ops.correlation1d_grad(gradients,
                                                       corr1d_op.inputs[0],
                                                       corr1d_op.inputs[1],
                                                       kernel_size,
                                                       max_displacement,
                                                       stride_1,
                                                       stride_2,
                                                       pad)
    return corr1d_grads.backpros_a, corr1d_grads.backpros_b


