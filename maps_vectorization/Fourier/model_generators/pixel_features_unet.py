import tensorflow as tf
from models_src.VecModels import SplitLayer, Vec2AngleActivationLayer
from models_src.UNet_model import UNet

def pixel_features_unet(
        input_shape, 
        init_filters_power, 
        levels, 
        level_convs, 
        init_dropout, 
        dropout, 
        batch_normalization, 
        name='PxFeaturesUnet', 
        **kwargs):
    
    
    unet_model = UNet(
        input_shape = input_shape,
        out_dims = 8,
        out_activation=None,
        init_filters_power=init_filters_power,
        levels=levels,
        level_convs=level_convs,
        color_embeddings=False,
        init_dropout=init_dropout,
        dropout=dropout,
        batch_normalization=batch_normalization,
        output_smoothing=False
    )

    inputs = unet_model.input
    out = unet_model.output

    out_shape_class, out_angle, out_thickness, out_center_vec = SplitLayer([3,2,1,2], name='Splits')(out)

    out_shape_class = tf.keras.layers.Activation('softmax', name='shape_class')(out_shape_class)
    out_angle = Vec2AngleActivationLayer(name='angle')(out_angle)
    #out_center_vec = CenterVecFormatter(name='CenterVec')(out_center_vec, out_angle)
    out_center_vec = tf.keras.layers.Identity(name='center_vec')(out_center_vec)
    out_thickness = tf.keras.layers.Identity(name='thickness')(out_thickness)
    return tf.keras.Model(inputs, {'shape_class': out_shape_class, 'angle': out_angle, 'thickness': out_thickness, 'center_vec': out_center_vec}, name=name)