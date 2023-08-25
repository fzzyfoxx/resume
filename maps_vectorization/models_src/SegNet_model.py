import tensorflow as tf
import math

class MaxPoolingArgMax(tf.keras.layers.Layer):
    def __init__(self, ksize=2, strides=2, padding='valid', **kwargs):
        super(MaxPoolingArgMax, self).__init__(**kwargs)
        self.ksize = ksize if type(ksize)==tuple else (ksize, ksize)
        self.strides = strides if type(strides)==tuple else (strides, strides)
        self.padding = padding.upper()

    def call(self, x):
        features, indices = tf.nn.max_pool_with_argmax(x, ksize=self.ksize, strides=self.strides, padding=self.padding, include_batch_in_index=False)
        return features, indices

    
    def compute_output_shape(self, input_shape):
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
    def __init__(self, size=(2,2), **kwargs):
        super().__init__(**kwargs)

        self.size = tf.constant((1,)+(size if type(size)==tuple else (size, size))+(1,), tf.int32)

    def build(self, input_shape):
        features_shape, indices_shape = input_shape[0], input_shape[1]
        self.indices_num = tf.reduce_prod(indices_shape[1:])

        self.scatter_output_size = tf.reduce_prod(self.size[1:]*features_shape[1:])
        self.features_size = tf.reduce_prod(features_shape[1:])

        self.output_features_shape = self.size[1:]*features_shape[1:]


    def call(self, inputs):
        features, indices = inputs[0], inputs[1]
        features_shape = tf.shape(features)
        batch_size = features_shape[0]
        indices = tf.concat([tf.repeat(tf.range(0,batch_size, dtype=tf.int64)[:,tf.newaxis, tf.newaxis], axis=-2, repeats=self.indices_num),
                               tf.reshape(indices, (batch_size, self.indices_num, 1))], axis=-1)
        
        output = tf.scatter_nd(tf.cast(indices, tf.int64), tf.reshape(features, (batch_size, self.features_size)), (batch_size, self.scatter_output_size))
        output = tf.reshape(output, tf.concat([tf.constant([-1]), self.output_features_shape], axis=0))
        return output

    def compute_output_shape(self, input_shape):
        return tf.cast(input_shape*self.size, tf.int32)
    

class SegNetConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), padding='same', pool_size=(2,2), conv_num=3, pooling=False, deconv=False, output_filters=None, **kwargs):
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
    def __init__(self, axis=-1, type='max', **kwargs):
        super().__init__(**kwargs)

        self.axis = axis

        if type=='max':
            self.aggr = tf.keras.backend.max
        elif type=='avg':
            self.aggr = tf.keras.backend.mean
        else:
            raise Exception('Wrong type. Try "max" or "avg"')

    def call(self, x):
        return self.aggr(x, axis=self.axis, keepdims=True)
        
    

class SegNet(tf.keras.Model):
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
                 aggregation_type='max',
                 **kwargs):
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
        
        self.aggr = ChannelPooling(type=aggregation_type)
        
        self.call(tf.keras.layers.Input(input_shape))

    def call(self, inputs):
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
        x = tf.keras.layers.Activation('sigmoid')(x)
        return x