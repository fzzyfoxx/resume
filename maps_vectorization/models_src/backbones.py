import tensorflow as tf
from models_src.DETR import FFN

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
    inputs = tf.keras.layers.Input(input_shape, name='IMG-Input')
    x = tf.keras.layers.Dense(filters, activation=activation, name='Input-Embeddings')(inputs)
    i=1
    for stage_length, strides in zip(stage_lengths, strides_list):
        x = gen_residual_stage(x, filters, f'Res-Stage-{i}',stage_length, block_length, kernel_size, activation, strides, batch_norm)
        i += 1
        filters *= 2

    return tf.keras.Model(inputs, x, name=name)

def gen_memorizing_res_backbone(filters, kernel_sizes=[3], block_length=2, add_pixel_conv=False, activation='relu', batch_norm=True, ffn_mid_layers=1, dropout=0.0, input_shape=(32,32,3), out_ind=None):
    inputs = tf.keras.layers.Input(input_shape, name='Patch-Input')
    x = FFN(mid_layers=ffn_mid_layers, mid_units=filters*2, output_units=filters, dropout=dropout, activation=activation, name='Color-Embeddings')(inputs)
    memory = [x]
    for i, kernel_size in enumerate(kernel_sizes):
        x = ResidualConvBlock(filters, block_length, kernel_size, activation, batch_norm=batch_norm, name=f'Res-Block-{i+1}')(x)
        memory.append(x)
        if add_pixel_conv:
            x = tf.keras.layers.Conv2D(filters, 1, activation='relu')(x)

    if out_ind is not None:
        memory = [m for i, m in enumerate(memory) if i in out_ind]
    out = tf.keras.layers.Concatenate(name='Channels-Concat')(memory)

    return tf.keras.Model(inputs, out)
