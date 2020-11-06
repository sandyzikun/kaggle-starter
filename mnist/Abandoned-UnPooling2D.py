# TODO: FIXME!
class UnPooling2D(keras.engine.base_layer.Layer):
    """2D Unpooling layer, which operates for spatial data.
    
    Arguments
    ===
        unpool_size: An integer or iterable (such as tuple or list)
            of 2 integers,
            which specifies the shape of expandation from one single
            element on the input tensor;
        data_format: A string,
            one of `"channels_first"` or `"channels_last"` (default).
            The ordering of the dimensions in the inputs.
            `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)` while `"channels_last"`
            corresponds to inputs with shape
            `(batch, height, width, channels)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you have never set it, then it will be `"channels_last"`;
    """
    def __init__(self, unpool_size=(2, 2), data_format=None, **kwargs):
        super(UnPooling2D, self).__init__(**kwargs)
        self.unpool_size = keras.utils.conv_utils.normalize_tuple(unpool_size, 2, "unpool_size")
        self.data_format = ((data_format if data_format in ["channels_first", "channels_last"] else None) or "channels_first").lower()

    def call(self, inputs):
        return

    def get_config(self):
        config = {
            "unpool_size": self.unpool_size,
            "data_format": self.data_format
            }
        return dict(list(super(UnPooling2D, self).get_config().items()) + list(config.items()))
# Abandoned

# TODO: FIXME!
def unpool2d(inputs):
    def _up2d(input_tensor, num_times=4):
        assert len(input_tensor.shape) == 2
        output_tensor = np.zeros(np.array(input_tensor.shape) * num_times)
        for i in range(output_tensor.shape[0]):
            for j in range(output_tensor.shape[1]):
                output_tensor[i, j] = input_tensor[i // num_times, j // num_times]
        return output_tensor
    output = np.array([ _up2d(inputs[idx]) for idx in range(len(inputs)) ])
    return output
# Abandoned

# TODO: FIXME!
UnPooling2D = keras.layers.core.Lambda(unpool2d, output_shape=(lambda input_shape: tuple(np.array(input_shape) * 2)))
# Testing