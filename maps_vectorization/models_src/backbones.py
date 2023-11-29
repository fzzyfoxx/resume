import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
class ResidualConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, length=2, kernel_size=3, activation='relu', strides=1, batch_norm=True, **kwargs):
        super(ResidualConvBlock, self).__init__(**kwargs)

        self.convs = tf.keras.Sequential(([tf.keras.layers.BatchNormalization()] if batch_norm else [])+[tf.keras.layers.Conv2D(filters, 
                                                                 kernel_size, 
                                                                 strides=(strides if i==0 else 1), 
                                                                 padding='same', 
                                                                 activation=(activation if i<length-1 else None),
                                                                 kernel_initializer='he_normal',
                                                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))
                                          for i in range(length)])
        
        if strides>1:
            self.skip_conv = tf.keras.layers.Conv2D(filters, 1, strides=strides, padding='same', activation=None, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.do_skip_conv = True if strides>1 else False

        self.out_activation = tf.keras.layers.Activation(activation)

    def call(self, inputs, training=None):
        memory = inputs
        x = self.convs(memory, training=training)
        if self.do_skip_conv:
            memory = self.skip_conv(memory, training=training)
        
        return self.out_activation(x + memory)


def gen_residual_stage(x, filters, name, stage_length=3, block_length=2, kernel_size=3, activation='relu', strides=2, batch_norm=True):
    for i in range(stage_length):
        x = ResidualConvBlock(filters, block_length, kernel_size, activation, (strides if i==0 else 1), batch_norm, name=f'{name}-{i+1}')(x)
    return x

def gen_backbone(filters, stage_lengths, strides_list, block_length=2, kernel_size=3, activation='relu', batch_norm=True, input_shape=(32,32,3), name='Backbone'):
    inputs = tf.keras.layers.Input(input_shape, dtype=tf.float32, name='IMG-Input')
    x = tf.keras.layers.Dense(filters, activation=activation, name='Input-Embeddings')(inputs)
    i=1
    for stage_length, strides in zip(stage_lengths, strides_list):
        x = gen_residual_stage(x, filters, f'Res-Stage-{i}',stage_length, block_length, kernel_size, activation, strides, batch_norm)
        i += 1
        filters *= 2

    return tf.keras.Model(inputs, x, name=name)