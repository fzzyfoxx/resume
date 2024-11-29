import tensorflow as tf
from models_src.VecModels import FrequencyRadialEncoding, SeparateRadialEncoding, YXcoordsLayer, RadialSearchFeaturesExtraction, SelfRadialMHA, SplitLayer, Vec2AngleActivationLayer, AddNorm
from models_src.DETR import FFN
from models_src.Attn_variations import SqueezeImg
from exp_lib.utils.load_mlflow_model import backbone_loader

def radial_enc_pixel_features_model_generator(
        enc_type,
        num_heads,
        embs_dim,
        color_embs_dim,
        size,
        embs_mid_layers,
        dropout,
        activation,
        out_mid_layers,
        attns_num,
        concat_memory,
        progressive,
        backbone_def,
        backbone_last_layer,
        backbone_init_layer,
        backbone_trainable,
        inverted_angle,
        name='PxFeaturesRadEnc'
):
    
    colors_only = embs_dim==color_embs_dim

    if colors_only:

        print('Colors only mode ON. Any backbone arguments provided are ignored.')
        if concat_memory:
            print('concat_memory set to False')
            concat_memory = False

        memory=None
        img_inputs = tf.keras.layers.Input((size, size, 3))
        normed_img = tf.keras.layers.BatchNormalization(name='Batch-Normalization')(img_inputs)
    else:
        backbone_model = backbone_loader(**backbone_def)

        backbone_model.trainable = backbone_trainable

        img_inputs = backbone_model.input
        memory = backbone_model.get_layer(backbone_last_layer).output
        normed_img = backbone_model.get_layer(backbone_init_layer).output

    
    #########
    enc_func = FrequencyRadialEncoding if enc_type!='separate' else SeparateRadialEncoding
    enc_label = 'Freq' if enc_type!='separate' else 'Sep'

    coords = YXcoordsLayer(size=(size,size), squeeze_output=True, name='Img-Coords')()

    pos_enc = enc_func(emb_dim=embs_dim//num_heads, height=size, inverted_angle=inverted_angle, name=f'{enc_label}RadialEncoding')(coords)

    features, pos_enc = RadialSearchFeaturesExtraction(embs_dim=embs_dim, 
                                                        color_embs_dim=color_embs_dim, 
                                                        mid_layers=embs_mid_layers,
                                                        activation=activation,
                                                        dropout=dropout,
                                                        batch_dims=1,
                                                        name='RSFE')(memory, normed_img, pos_enc)

    print(pos_enc.shape, features.shape)

    for i in range(attns_num):
        #V = tf.keras.layers.Permute([2,1,3], name=f'PreMHA-Permute_{i+1}')(features)
        i_heads = num_heads*2**i if progressive else num_heads
        i_embs = embs_dim*2**i if progressive else embs_dim

        x, _ = SelfRadialMHA(output_dim=i_embs, value_dim=i_embs, key_dim=i_embs, num_heads=i_heads, name=f'MHA_{i+1}')(pos_enc, features, features)
        #print(x.shape)
        if i>0:
            if progressive:
                features = FFN(mid_layers=out_mid_layers, mid_units=i_embs, output_units=i_embs, dropout=dropout, activation=activation, name=f'Progressive-SkipCon-FFN_{i+1}')(features)
            features = AddNorm(norm_axis=-1, name=f'PostMHA-AddNorm_{i+1}')([features, x])
            x = FFN(mid_layers=out_mid_layers, mid_units=i_embs*2, output_units=i_embs, dropout=0.0, activation=activation, name=f'Decoder-FFN_{i+1}')(features)
            features = AddNorm(norm_axis=-1, name=f'PostFFN-AddNorm_{i+1}')([features, x])
        else:
            features = FFN(mid_layers=out_mid_layers, mid_units=i_embs*2, output_units=i_embs, dropout=0.0, activation=activation, name=f'Decoder-FFN_{i+1}')(x)
    
    if concat_memory:
        memory = SqueezeImg(name='Squeeze-Memory')(memory)
        features = tf.keras.layers.Concatenate(axis=-1, name='Concat-Memory')([memory, features])
        
    out = FFN(mid_layers=out_mid_layers, mid_units=i_embs*2, output_units=8, dropout=dropout, activation=activation, name=f'Out-FFN')(features)
    out = tf.keras.layers.Reshape((size, size, 8))(out)

    out_shape_class, out_angle, out_thickness, out_center_vec = SplitLayer([3,2,1,2], name='Splits')(out)

    out_shape_class = tf.keras.layers.Activation('softmax', name='shape_class')(out_shape_class)
    out_angle = Vec2AngleActivationLayer(name='angle')(out_angle)
    out_center_vec = tf.keras.layers.Identity(name='center_vec')(out_center_vec)
    out_thickness = tf.keras.layers.Identity(name='thickness')(out_thickness)
    model = tf.keras.Model(img_inputs, {'shape_class': out_shape_class, 'angle': out_angle, 'thickness': out_thickness, 'center_vec': out_center_vec}, name=name)
    
    return model