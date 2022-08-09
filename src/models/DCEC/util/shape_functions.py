def compute_output_shape(current_shape, kernel_size, stride, padding):
    """
    Output shape obtained from convolutional layer.
    :param tuple current_shape:  shape current prima dell'applicazione della convoluzione
    :param tuple kernel_size:    Grandezza del kernel
    :param tuple stride:         Stride.
    :param tuple padding:        Padding

    :return:  Shape dopo aver applicato la convoluzione.
    :rtype:   tuple


        component[i] = floor((N[i] - K[i] + 2 * P[i]) / S[i]) + 1

        where, N = current shape of the data
               K = kernel size
               P = padding
               S = stride
    """
    # get the dimension of the data compute each component using the above formula
    dimensions = len(current_shape)
    return tuple((current_shape[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1 for i in range(dimensions))


def compute_transpose_output_shape(current_shape, kernel_size, stride, padding):
    """
    Output shape obtained from transopose convolutional layer.
    :param tuple current_shape:  The current shape of the data before a transpose convolution is
                                   applied.
    :param tuple kernel_size:    The kernel size of the current transpose convolution operation.
    :param tuple stride:         The stride of the current transpose convolution operation.
    :param tuple padding:        The padding of the current transpose convolution operation.

    :return:  The shape after a transpose convolution operation with the above parameters is
                applied.
    :rtype:   tuple

            The formula used to compute the final shape is

        component[i] = (N[i] - 1) * S[i] - 2 * P[i] + (K[i] - 1) + 1

        where, N = current shape of the data
               K = kernel size
               P = padding
               S = stride
    """
    # get the dimension of the data compute each component using the above formula
    dimensions = len(current_shape)
    return tuple(
        (current_shape[i] - 1) * stride[i] - 2 * padding[i] + (kernel_size[i] - 1) + 1 for i in range(dimensions))


def compute_output_padding(current_shape, target_shape):
    """
    This function computes the outpout-padding to apply in a de-conv layer application oin order to obtain a defined
    target shape.
    :param tuple current_shape:  The shape of the data after a transpose convolution operation
                                   takes place.
    :param tuple target_shape:   The target shape that we would like our data to have after the
                                   transpose convolution operation takes place.

    :return:  The output padding needed so that the shape of the image after a transpose
                convolution is applied, is the same as the target shape.
    :rtype:   tuple
    """
    # basically subtract each term to get the difference which will be the output padding
    dimensions = len(current_shape)
    return tuple(target_shape[i] - current_shape[i] for i in range(dimensions))

