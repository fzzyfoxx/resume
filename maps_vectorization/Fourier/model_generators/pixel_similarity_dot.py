import tensorflow as tf
from models_src.Attn_variations import SqueezeImg
from models_src.DETR import FFN, HeadsPermuter
from models_src.VecModels import DotSimilarityLayer
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