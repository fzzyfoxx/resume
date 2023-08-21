import tensorflow as tf

class UNetConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, padding, pooling_size, dropout, **kwargs):
        super().__init__(**kwargs)

        self.conv_layers = [tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding) for _ in range(2)]
        self.MaxPool = tf.keras.layers.MaxPooling2D(pooling_size)
        self.Dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
        
        x_pool = self.MaxPool(x)
        return self.Dropout(x_pool, training=training), x
    
class UNetUpConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, padding, strides, dropout, **kwargs):
        super().__init__(**kwargs)

        self.conv_transpose = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, activation=activation, padding=padding)
        self.conv_layers = [tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, padding=padding) for _ in range(2)]
        self.Dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shapes):
        x_shape = input_shapes[0][1:-1]
        saved_x_shape = input_shapes[1][1:-1]

        padding = tuple([(0,b-2*a) for a,b in zip(x_shape, saved_x_shape)])

        self.pad = tf.keras.layers.ZeroPadding2D(padding)

    def call(self, inputs, training=False):
        x = inputs[0]
        saved_x = inputs[1]

        x = self.conv_transpose(x)
        x = self.pad(x)
        x = tf.keras.layers.Concatenate()([x, saved_x])
        x = self.Dropout(x, training=training)
        for conv in self.conv_layers:
            x = conv(x)

        return x

class UNet(tf.keras.Model):
    def __init__(self, init_filters_power=6, levels=5, kernel_size=(3,3), pooling_size=(2,2), init_dropout=0.25, dropout=0.5, padding='same', **kwargs):
        super(UNet, self).__init__(self, **kwargs)
        conv2D_args = {'kernel_size': kernel_size, 'activation': 'relu', 'padding': padding}

        conv_dropout_list = [init_dropout]+[dropout]*(levels-1)
        upconv_dropout_list = [dropout]*(levels-1)

        init_filters = 2**init_filters_power

        # downsize layers
        self.downsize_layers = [UNetConvBlock(init_filters*2**i, **conv2D_args, pooling_size=pooling_size, dropout=d) for i,d in enumerate(conv_dropout_list)]
        self.upsize_layers = [UNetUpConvBlock(init_filters*2**(levels-i-2), **conv2D_args, strides=pooling_size, dropout=d) for i,d in enumerate(upconv_dropout_list)]

        self.final_conv = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')
    
    def call(self, inputs, training=False):
        conv_levels = []
        x = inputs

        # downsize part
        for conv in self.downsize_layers:
            x, x_saved = conv(x, training=training)
            conv_levels.append(x_saved)

        # upsize part
        x = x_saved # last convolution output (without pooling)

        # iterate descending over convolution outputs without the last one
        for x_saved, upconv in zip(conv_levels[:-1][::-1], self.upsize_layers):
            x = upconv([x, x_saved])

        return self.final_conv(x)
    
class SmoothOutput(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(SmoothOutput, self).__init__(**kwargs)

        self.epsilon = epsilon

    def call(self, x, training=False):
        x = tf.clip_by_value(x, self.epsilon, 1-self.epsilon)
        return x