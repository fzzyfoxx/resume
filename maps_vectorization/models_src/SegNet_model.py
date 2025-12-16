import tensorflow as tf
import math

class MaxPoolingArgMax(tf.keras.layers.Layer):
    """Max-pooling layer that also returns the argmax indices.

    This layer performs a max pooling operation and returns both the
    pooled features and the indices of the max values (argmax).

    Attributes:
        ksize: Pooling window size as an int or a tuple (h, w).
        strides: Stride of the pooling as an int or tuple (h, w).
        padding: Padding mode, either 'VALID' or 'SAME'.
    """
    def __init__(self, ksize=2, strides=2, padding='valid', **kwargs):
        """Initialize the MaxPoolingArgMax layer.

        Args:
            ksize: Size of the pooling window (int or tuple).
            strides: Stride for the pooling (int or tuple).
            padding: Padding mode string, 'valid' or 'same'.
            **kwargs: Additional keyword arguments passed to the base Layer.
        """
        super(MaxPoolingArgMax, self).__init__(**kwargs)
        self.ksize = ksize if type(ksize)==tuple else (ksize, ksize)
        self.strides = strides if type(strides)==tuple else (strides, strides)
        self.padding = padding.upper()

    def call(self, x):
        """Apply max pooling and return pooled features and argmax indices.

        Args:
            x: Input tensor of shape (batch, height, width, channels).

        Returns:
            A tuple (pooled_features, argmax_indices). The argmax indices can be
            used later to unpool with the corresponding upsampling layer.
        """
        features, indices = tf.nn.max_pool_with_argmax(x, ksize=self.ksize, strides=self.strides, padding=self.padding, include_batch_in_index=False)
        return features, indices

    
    def compute_output_shape(self, input_shape):
        """Compute the output shape of the pooling operation.

        Args:
            input_shape: Shape tuple (batch, height, width, channels).

        Returns:
            A list containing two shape tuples: the pooled features shape and
            the indices shape (same as features shape).
        """
        b,h,w,c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        size_vec = [h,w]
        strides_vec = self.strides

        if self.padding=='SAME':
            output_vec = [math.ceil(size/stride) for size, stride in zip(size_vec, strides_vec)]
        elif self.padding=='VALID':
            output_vec = [math.floor((size-1)/stride+1) for size, stride in zip(size_vec, strides_vec)]

        output_shape = tuple([b]+output_vec+[c])

        return [output_shape, output_shape]

    
class ArgMaxUpsample(tf.keras.layers.Layer):
    """Upsampling layer that uses previously computed argmax indices to place values.

    This layer reverses MaxPoolingArgMax by scattering the pooled feature values
    back to their original spatial locations using saved argmax indices.

    Attributes:
        size: Upsampling factor as a tuple (h_factor, w_factor).
    """
    def __init__(self, size=(2,2), **kwargs):
        """Initialize the ArgMaxUpsample layer.

        Args:
            size: Upsampling multiplier (int or tuple).
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

        self.size = tf.constant((1,)+(size if type(size)==tuple else (size, size))+(1,), tf.int32)

    def build(self, input_shape):
        """Prepare internal sizes and counters based on input shapes.

        Args:
            input_shape: A tuple or list with two shapes: features and indices.
        """
        features_shape, indices_shape = input_shape[0], input_shape[1]
        self.indices_num = tf.reduce_prod(indices_shape[1:])

        self.scatter_output_size = tf.reduce_prod(self.size[1:]*features_shape[1:])
        self.features_size = tf.reduce_prod(features_shape[1:])

        self.output_features_shape = self.size[1:]*features_shape[1:]


    def call(self, inputs):
        """Scatter the pooled feature values back into their upsampled positions.

        Args:
            inputs: A tuple (features, indices) where features is the pooled
                    tensor and indices are argmax positions from MaxPoolingArgMax.

        Returns:
            A tensor with upsampled spatial dimensions.
        """
        features, indices = inputs[0], inputs[1]
        features_shape = tf.shape(features)
        batch_size = features_shape[0]
        indices = tf.concat([tf.repeat(tf.range(0,batch_size, dtype=tf.int64)[:,tf.newaxis, tf.newaxis], axis=-2, repeats=self.indices_num),
                               tf.reshape(indices, (batch_size, self.indices_num, 1))], axis=-1)
        
        output = tf.scatter_nd(tf.cast(indices, tf.int64), tf.reshape(features, (batch_size, self.features_size)), (batch_size, self.scatter_output_size))
        output = tf.reshape(output, tf.concat([tf.constant([-1]), self.output_features_shape], axis=0))
        return output

    def compute_output_shape(self, input_shape):
        """Compute the output shape after upsampling.

        Args:
            input_shape: Shapes of the input tensors.

        Returns:
            The upsampled shape as a tf.Tensor of integers.
        """
        return tf.cast(input_shape*self.size, tf.int32)
    

class SegNetConv(tf.keras.layers.Layer):
    """Convolutional block used in the SegNet encoder and decoder.

    This block can perform a sequence of convolutions followed by batch
    normalization and ReLU activation. It optionally applies pooling (encoder)
    or argmax-based unpooling (decoder).

    Args:
        filters: Number of convolutional filters for the block.
        kernel_size: Convolution kernel size.
        padding: Padding mode for convolution.
        pool_size: Pooling size used for max-pooling/unpooling.
        conv_num: Number of convolution layers in the block.
        pooling: If True, apply MaxPoolingArgMax at the end (encoder behavior).
        deconv: If True, perform ArgMaxUpsample before convolutions (decoder).
        output_filters: If provided, use this number of filters for the last conv.
    """
    def __init__(self, filters, kernel_size=(3,3), padding='same', pool_size=(2,2), conv_num=3, pooling=False, deconv=False, output_filters=None, **kwargs):
        """Initialize a SegNet convolutional block.

        All arguments are forwarded to configure internal layers and behavior.
        """
        super().__init__(**kwargs)

        self.pooling = pooling
        self.deconv = deconv

        output_filters = output_filters if output_filters else filters

        self.convs = [tf.keras.layers.Conv2D(filters, kernel_size, padding=padding) for _ in range(conv_num-1)] + \
                        [tf.keras.layers.Conv2D(output_filters, kernel_size, padding=padding)]
        self.max_pooling = MaxPoolingArgMax(ksize=pool_size, strides=pool_size, padding=padding)
        self.deconv_layer = ArgMaxUpsample(size=pool_size)

        self.norms = [tf.keras.layers.BatchNormalization() for _ in range(conv_num)]
        self.activations = [tf.keras.activations.relu for _ in range(conv_num)]

    def call(self, x):
        """Apply the convolutional sequence and optional pooling/deconvolution.

        Args:
            x: If used in encoder mode, a tensor input. If used in decoder mode
               (deconv=True), x is expected to be a tuple/list (features, indices)
               so the deconv_layer can perform unpooling first.

        Returns:
            If pooling=True (encoder): a tuple (pooled_features, pool_indices).
            Otherwise: the processed tensor.
        """

        if self.deconv:
            x = self.deconv_layer(x)

        for conv, norm, activation in zip(self.convs, self.norms, self.activations):
            x = conv(x)
            x = norm(x)
            x = activation(x)

        if self.pooling:
            x = self.max_pooling(x)

        return x

class ChannelPooling(tf.keras.layers.Layer):
    """Simple channel-wise pooling layer to reduce channel dimension.

    Supports taking either the channel-wise maximum or average and retains the
    channel dimension as 1.

    Args:
        axis: Axis along which to aggregate (default -1 for channels).
        type: Aggregation type: 'max' or 'avg'.
    """
    def __init__(self, axis=-1, type='max', **kwargs):
        """Initialize the ChannelPooling layer.

        Args:
            axis: Axis to aggregate across (default -1 channels).
            type: 'max' or 'avg' aggregation.
        """
        super().__init__(**kwargs)

        self.axis = axis

        if type=='max':
            self.aggr = tf.keras.backend.max
        elif type=='avg':
            self.aggr = tf.keras.backend.mean
        else:
            raise Exception('Wrong type. Try "max" or "avg"')

    def call(self, x):
        """Aggregate input tensor along the specified channel axis.

        Args:
            x: Input tensor.

        Returns:
            Aggregated tensor with the same rank but reduced channel dimension.
        """
        return self.aggr(x, axis=self.axis, keepdims=True)
        
    

class SegNet(tf.keras.Model):
    """SegNet-style encoder-decoder model built from SegNetConv blocks.

    This implementation builds symmetric encoder and decoder stacks using
    argmax-based pooling and unpooling. The final aggregation can be a small
    convolution or channel pooling followed by an activation.

    Args:
        input_shape: Shape of the model input (height, width, channels).
        init_filters_power: Base exponent for number of filters (2**power).
        levels: Number of encoder/decoder levels.
        conv_per_level: Number of conv layers per standard level.
        shorten_levels: Number of initial levels with fewer convs.
        conv_per_shorten_level: Conv count for shortened levels.
        kernel_size: Convolution kernel size.
        padding: Padding mode for convolutions and pooling.
        pool_size: Pooling/unpooling size.
        aggregation_type: If None, use a Conv2D to combine channels; otherwise use ChannelPooling.
        output_filters: Number of filters in the aggregation layer or output channels.
        output_activation: Activation applied to final outputs.
    """
    def __init__(self, 
                 input_shape=(256,256,3),
                 init_filters_power=6, 
                 levels=5, 
                 conv_per_level=3, 
                 shorten_levels=2, 
                 conv_per_shorten_level=2, 
                 kernel_size=(3,3), 
                 padding='same', 
                 pool_size=(2,2), 
                 aggregation_type=None,
                 output_filters=1,
                 output_activation='sigmoid',
                 **kwargs):
        """Construct the SegNet model and build its layers.

        The constructor also performs an initial call with a dummy input to
        force layer building and shape inference.
        """
        super(SegNet, self).__init__(self, **kwargs)

        convs_schema = [conv_per_shorten_level for _ in range(shorten_levels)] + [conv_per_level for _ in range(levels-shorten_levels)]

        self.encoder_convs = [SegNetConv(filters=2**(init_filters_power+n),
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                pool_size=pool_size,
                                                conv_num=conv_num,
                                                pooling=True,
                                                name='EncoderConv-'+str(n)
                                                ) for n, conv_num in enumerate(convs_schema)]
        
        self.decoder_convs = [SegNetConv(filters=2**(init_filters_power+n),
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                pool_size=pool_size,
                                                conv_num=conv_num,
                                                deconv=True,
                                                output_filters=2**(init_filters_power+n-(1 if n>0 else 0)),
                                                name='DecoderConv-'+str(n)
                                                ) for n, conv_num in enumerate(convs_schema)][::-1]
        
        self.aggr = tf.keras.layers.Conv2D(output_filters, 1, padding='same') if aggregation_type==None else ChannelPooling(type=aggregation_type)

        self.output_activation = tf.keras.layers.Activation(output_activation)
        
        self.call(tf.keras.layers.Input(input_shape))

    def call(self, inputs):
        """Run a forward pass through the encoder, decoder and aggregation.

        Args:
            inputs: Input tensor to the model.

        Returns:
            Output tensor after decoding, aggregation and activation.
        """
        pool_idxs_list = []

        x = inputs
        # Encoder
        for encoder_conv in self.encoder_convs:
            x, pool_idxs = encoder_conv(x)
            pool_idxs_list.append(pool_idxs)

        #Decoder
        for decoder_conv, pool_idxs in zip(self.decoder_convs, pool_idxs_list[::-1]):
            x = decoder_conv([x, pool_idxs])

        x = self.aggr(x)
 
        x = self.output_activation(x)
        return x