import tensorflow as tf
from models_src.Attn_variations import SqueezeImg, UnSqueezeImg
from models_src.DETR import FFN, HeadsPermuter
from models_src.VecModels import DotSimilarityLayer, FrequencyRadialEncoding, SeparateRadialEncoding, YXcoordsLayer, RadialSearchFeaturesExtraction, SelfRadialMHA, SplitLayer, Vec2AngleActivationLayer, AddNorm
from exp_lib.utils.load_mlflow_model import backbone_loader

def backbone_based_pixel_similarity_dot_model(
        color_embs_num,
        color_embs_mid_layers,
        conv_num,
        conv_dim,
        attn_dim,
        heads_num,
        use_heads,
        pre_attn_ffn_mid_layers,
        dropout,
        backbone_def,
        backbone_last_layer,
        backbone_init_layer,
        backbone_trainable,
        name='PxSimDot'
        ):
    
    conv_dim = attn_dim//2
    
    backbone_model = backbone_loader(**backbone_def)

    backbone_model.trainable = backbone_trainable
    inputs = backbone_model.input
    memory = backbone_model.get_layer(backbone_last_layer).output
    normed_img = backbone_model.get_layer(backbone_init_layer).output

    normed_img = SqueezeImg(name='Squeeze-NormedImg')(normed_img)
    color_embs_map = FFN(mid_layers=color_embs_mid_layers, mid_units=color_embs_num*2, output_units=color_embs_num, dropout=dropout, activation='relu', name='Colors_FFN')(normed_img)

    if conv_num>0:
        memory = tf.keras.layers.Conv2D(conv_dim, kernel_size=1, padding='same', activation='relu', name='Conv_init')(memory)
        x = memory
        for i in range(conv_num):
            x = tf.keras.layers.Conv2D(conv_dim, kernel_size=3, padding='same', activation='relu', name=f'Conv_{i+1}')(x)

        x = tf.keras.layers.Concatenate(name='Concat_Memory')([x, memory])
    else:
        x = memory
    x = SqueezeImg(name='Squeeze-Memory')(x)
    x = FFN(mid_layers=pre_attn_ffn_mid_layers, mid_units=attn_dim*2, output_units=attn_dim, dropout=dropout, activation='relu', name='Features_FFN')(x)
    x = tf.keras.layers.Concatenate(name='Concat_Color_Embs')([x, color_embs_map])

    if use_heads:
        x = HeadsPermuter(num_heads=heads_num, emb_dim=(attn_dim+color_embs_num)//heads_num, name='Heads_Permute')(x)
        x = tf.keras.layers.LayerNormalization(axis=-1, name='Heads_Norm')(x)
        x = HeadsPermuter(num_heads=heads_num, emb_dim=(attn_dim+color_embs_num)//heads_num, reverse=True, name='Heads_Unpermute')(x)
    else:
        x = tf.keras.layers.LayerNormalization(axis=-1, name='Norm')(x)
    out = DotSimilarityLayer(epsilon=1e-6, name='Dot_Similarity')(x)

    return tf.keras.Model(inputs, {'Dot_Similarity': out}, name=name)


def radial_enc_pixel_similarity_dot(
        color_embs_num,
        color_embs_mid_layers,
        enc_type,
        num_heads,
        embs_dim,
        color_embs_dim,
        size,
        embs_mid_layers,
        dropout,
        activation,
        attn_mid_layers,
        out_mid_layers,
        attns_num,
        attn_concat_colors,
        concat_memory,
        concat_colors,
        progressive,
        inverted_angle,
        backbone_def,
        backbone_last_layer,
        backbone_init_layer,
        backbone_trainable,
        name='PxRadSimDot'
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

    color_embs_map = FFN(mid_layers=color_embs_mid_layers, mid_units=color_embs_num*2, output_units=color_embs_num, dropout=dropout, activation='relu', name='Colors_FFN')(normed_img)

    features, pos_enc = RadialSearchFeaturesExtraction(embs_dim=embs_dim, 
                                                        color_embs_dim=color_embs_dim, 
                                                        mid_layers=embs_mid_layers,
                                                        activation=activation,
                                                        dropout=dropout,
                                                        batch_dims=1,
                                                        name='RSFE')(memory, color_embs_map, pos_enc)
    
    if attn_concat_colors | concat_colors:
        color_embs_map = SqueezeImg(name='Squeeze-Colors')(color_embs_map)

    print(pos_enc.shape, features.shape)

    for i in range(attns_num):
        #V = tf.keras.layers.Permute([2,1,3], name=f'PreMHA-Permute_{i+1}')(features)
        i_heads = num_heads*2**i if progressive else num_heads
        i_embs = embs_dim*2**i if progressive else embs_dim

        x, _ = SelfRadialMHA(output_dim=i_embs, value_dim=i_embs, key_dim=i_embs, num_heads=i_heads, name=f'MHA_{i+1}')(pos_enc, features, features)
        
        if progressive:
            features = FFN(mid_layers=attn_mid_layers, mid_units=i_embs, output_units=i_embs, dropout=dropout, activation=activation, name=f'Progressive-SkipCon-FFN_{i+1}')(features)
        features = AddNorm(norm_axis=-1, name=f'PostMHA-AddNorm_{i+1}')([features, x])

        x = tf.keras.layers.Concatenate(axis=-1, name=f'PostMHA-Concat-Colors_{i+1}')([color_embs_map, features]) if attn_concat_colors else features

        x = FFN(mid_layers=attn_mid_layers, mid_units=i_embs*2, output_units=i_embs, dropout=0.0, activation=activation, name=f'Decoder-FFN_{i+1}')(x)
        features = AddNorm(norm_axis=-1, name=f'PostFFN-AddNorm_{i+1}')([features, x])
    
    if concat_memory:
        memory = SqueezeImg(name='Squeeze-Memory')(memory)
        features = tf.keras.layers.Concatenate(axis=-1, name='Concat-Memory')([memory, features])

    if concat_colors:
        features = tf.keras.layers.Concatenate(axis=-1, name='Concat-Colors')([color_embs_map, features])
    
    out_emb_size = features.shape[-1]
    Q = FFN(mid_layers=out_mid_layers, mid_units=out_emb_size*2, output_units=out_emb_size, dropout=0.0, activation=activation, name=f'Out-Q-FFN')(features)
    K = FFN(mid_layers=out_mid_layers, mid_units=out_emb_size*2, output_units=out_emb_size, dropout=0.0, activation=activation, name=f'Out-K-FFN')(features)
    out = DotSimilarityLayer(epsilon=1e-6, name='Dot_Similarity')(Q, K)

    return tf.keras.Model(img_inputs, {'Dot_Similarity': out}, name=name)