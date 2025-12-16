import tensorflow as tf
from models_src.DETR import FFN

"""Backbone utilities for convolutional residual encoders.

This module provides a ResidualConvBlock layer and helper functions to
assemble multi-stage convolutional backbones suitable for image or patch
embeddings. It also includes a small memorizing backbone that returns
intermediate feature maps (memory) for use in later modules.
"""

@tf.keras.saving.register_keras_serializable()
class ResidualConvBlock(tf.keras.layers.Layer):
    """Residual convolutional block with optional downsampling.

    This block stacks one or more Conv2D layers (optionally preceded by
    BatchNormalization) and adds a residual skip connection. When
    strides>1, the skip connection is projected with a 1x1 convolution
    to match spatial and channel dimensions.

    Args:
        filters (int): Number of convolutional filters for each conv layer.
        length (int): Number of convolutional layers inside the block.
        kernel_size (int or tuple): Kernel size to use for Conv2D layers.
        activation (str or callable): Activation applied to intermediate
            layers and the block output.
        strides (int): Stride for the first convolution in the block. If
            >1, the block performs spatial downsampling and the skip
            connection is projected accordingly.
        batch_norm (bool): If True, apply a BatchNormalization layer before
            the convolutions.
        **kwargs: Additional keyword arguments passed to the Layer base.

    Example:
        block = ResidualConvBlock(64, length=2, kernel_size=3, strides=2)
        out = block(inputs)
    """
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
        """Forward pass for the residual block.

        Applies the sequential convolutional operations to the input and
        adds the (possibly projected) skip connection. The final
        activation is applied to the elementwise sum.

        Args:
            inputs (Tensor): Input feature map of shape [batch, H, W, C].
            training (bool, optional): Whether the layer should behave in
                training mode (affects BatchNormalization).

        Returns:
            Tensor: Output feature map after residual addition and
            activation, with spatial dimensions possibly reduced when
            strides>1.
        """
        memory = inputs
        x = self.convs(memory, training=training)
        if self.do_skip_conv:
            memory = self.skip_conv(memory, training=training)
        
        return self.out_activation(x + memory)


def gen_residual_stage(x, filters, name, stage_length=3, block_length=2, kernel_size=3, activation='relu', strides=2, batch_norm=True):
    """Generate a stage composed of multiple ResidualConvBlock instances.

    A stage consists of `stage_length` ResidualConvBlock blocks. The
    first block in the stage uses the provided `strides` value (allowing
    downsampling), while subsequent blocks use stride 1.

    Args:
        x (Tensor): Input tensor to the stage.
        filters (int): Number of filters for blocks in this stage.
        name (str): Base name used for child blocks.
        stage_length (int): Number of blocks in the stage.
        block_length (int): Number of conv layers inside each block.
        kernel_size (int or tuple): Kernel size for conv layers.
        activation (str or callable): Activation used inside blocks.
        strides (int): Stride for the first block in the stage.
        batch_norm (bool): Whether to include BatchNormalization.

    Returns:
        Tensor: Output tensor after applying the stage.
    """
    for i in range(stage_length):
        x = ResidualConvBlock(filters, block_length, kernel_size, activation, (strides if i==0 else 1), batch_norm, name=f'{name}-{i+1}')(x)
    return x

def gen_backbone(filters, stage_lengths, strides_list, block_length=2, kernel_size=3, activation='relu', batch_norm=True, input_shape=(32,32,3), name='Backbone'):
    """Construct a sequential residual backbone model.

    The backbone starts with a Dense projection of the input and then
    stacks stages generated by `gen_residual_stage`. After each stage the
    number of filters is doubled.

    Args:
        filters (int): Number of filters for the first stage.
        stage_lengths (iterable): Sequence of integers specifying the
            number of blocks per stage.
        strides_list (iterable): Sequence of stride values for each
            corresponding stage (used for the first block in the stage).
        block_length (int): Number of conv layers inside each ResidualConvBlock.
        kernel_size (int or tuple): Kernel size for conv layers.
        activation (str or callable): Activation used throughout the backbone.
        batch_norm (bool): Whether to include BatchNormalization layers.
        input_shape (tuple): Shape of the input tensor (H, W, C).
        name (str): Name for the returned Keras Model.

    Returns:
        tf.keras.Model: A model mapping input images to the final stage
        feature map tensor.
    """
    inputs = tf.keras.layers.Input(input_shape, name='IMG-Input')
    x = tf.keras.layers.Dense(filters, activation=activation, name='Input-Embeddings')(inputs)
    i=1
    for stage_length, strides in zip(stage_lengths, strides_list):
        x = gen_residual_stage(x, filters, f'Res-Stage-{i}',stage_length, block_length, kernel_size, activation, strides, batch_norm)
        i += 1
        filters *= 2

    return tf.keras.Model(inputs, x, name=name)

def gen_memorizing_res_backbone(filters, kernel_sizes=[3], block_length=2, add_pixel_conv=False, activation='relu', batch_norm=True, ffn_mid_layers=1, dropout=0.0, input_shape=(32,32,3), out_ind=None):
    """Build a patch-level backbone that returns intermediate feature memories.

    This function creates a small network that begins with a feed-forward
    network (FFN) to embed the input patch and then applies a sequence of
    ResidualConvBlock layers. Intermediate outputs are collected into a
    memory list and concatenated at the end. Optionally, only a subset
    of memory indices can be selected via `out_ind`.

    Args:
        filters (int): Number of filters for ResidualConvBlock layers.
        kernel_sizes (list[int]): List of kernel sizes, one per block.
        block_length (int): Number of conv layers inside each ResidualConvBlock.
        add_pixel_conv (bool): If True, add a 1x1 Conv2D after each block.
        activation (str or callable): Activation function to use.
        batch_norm (bool): Whether to use BatchNormalization in blocks.
        ffn_mid_layers (int): Number of mid layers in the initial FFN.
        dropout (float): Dropout rate applied in the FFN.
        input_shape (tuple): Shape of the input patch (H, W, C).
        out_ind (iterable or None): Optional indices of memory entries to
            keep before concatenation. If None, all memories are used.

    Returns:
        tf.keras.Model: A model mapping input patches to a concatenated
        tensor of selected intermediate feature maps.
    """
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
