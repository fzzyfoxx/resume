import tensorflow as tf
import models_src.VecModels as vcm
from models_src.Attn_variations import UnSqueezeImg, SqueezeImg
from models_src.DETR import FFN
from exp_lib.utils.load_mlflow_model import backbone_loader

def radial_enc_vec_detection_model_generator(
        enc_type,
        num_heads,
        embs_dim,
        color_embs_dim,
        size,
        sample_points,
        embs_mid_layers,
        dropout,
        activation,
        out_mid_layers,
        attns_num,
        pos_enc_matmul,
        queries_self_attn,
        queries_pos_enc_values,
        source_query_cross_attn,
        sample_queries_num,
        angle_based_preds,
        backbone_def,
        backbone_last_layer,
        backbone_init_layer,
        backbone_trainable,
        features_update,
        thickness_pred,
        name='RadVecDet'
):
    
    backbone_model = backbone_loader(**backbone_def)

    backbone_model.trainable = backbone_trainable

    img_inputs = backbone_model.input
    memory = backbone_model.get_layer(backbone_last_layer).output
    normed_img = backbone_model.get_layer(backbone_init_layer).output

    if len(memory.shape)==3:
        memory = UnSqueezeImg(name='Initial-Memory-Unsqueeze')(memory)

    sample_inputs = tf.keras.layers.Input((sample_points,2), name='sample_points')
    #split_inputs = tf.keras.layers.Input((cfg.sample_points,), name='class_split')

    enc_func = vcm.FrequencyRadialEncoding if enc_type!='separate' else vcm.SeparateRadialEncoding
    self_enc_func = vcm.SampleFrequencyRadialEncoding if enc_type!='separate' else vcm.SampleSeparateRadialEncoding
    enc_label = 'Freq' if enc_type!='separate' else 'Sep'

    pos_enc = enc_func(emb_dim=embs_dim//num_heads, height=size, name=f'{enc_label}RadialEncoding')(sample_inputs)
    if queries_self_attn:
        self_pos_enc = vcm.SampleFrequencyRadialEncoding(emb_dim=embs_dim//num_heads, height=size, name=f'Self{enc_label}RadialEncoding')(sample_inputs, sample_inputs)

    features, sample_features, pos_enc = vcm.SampleRadialSearchFeaturesExtraction(embs_dim=embs_dim, 
                                                                            color_embs_dim=color_embs_dim, 
                                                                            mid_layers=embs_mid_layers,
                                                                            activation=activation,
                                                                            dropout=dropout,
                                                                            batch_dims=1,
                                                                            name='vdSRSFE')(sample_inputs, memory, normed_img, pos_enc)


    value_pos_enc = not pos_enc_matmul

    for i in range(attns_num):


        x, weights, scores = vcm.DetectionMHA(key_pos_enc=(i!=0), pos_enc_matmul=pos_enc_matmul, value_pos_enc=value_pos_enc, output_dim=embs_dim, value_dim=embs_dim, key_dim=embs_dim, num_heads=num_heads, return_scores=True, name=f'vdMHA_{i+1}')(features, sample_features, features, pos_enc)
        #print(x.shape)
        sample_features = vcm.AddNorm(norm_axis=-1, name=f'vdPostMHA-AddNorm_{i+1}')([sample_features, x])

        if queries_self_attn:
            ### sample self-attn
            V = self_pos_enc if queries_pos_enc_values else tf.keras.layers.Permute([2,1,3], name=f'PreSelfMHA-Permute_{i+1}')(x)     
            x, _ = vcm.DetectionMHA(key_pos_enc=True, value_pos_enc=False, output_dim=embs_dim, value_dim=embs_dim, key_dim=embs_dim, num_heads=num_heads, name=f'SelfMHA_{i+1}')(V, sample_features, V, self_pos_enc)
            sample_features = vcm.AddNorm(norm_axis=-1, name=f'PostSelfMHA-AddNorm_{i+1}')([sample_features, x])

        if source_query_cross_attn & (i<attns_num-1):
            Q_pos_enc = tf.keras.layers.Permute([2,1,3], name=f'PreCrossMHA-PosEnc-Permute_{i+1}')(pos_enc)
            V = tf.keras.layers.Permute([2,1,3], name=f'PreCrossMHA-Sample-Permute_{i+1}')(sample_features)
            Q = tf.keras.layers.Permute([2,1,3], name=f'PreCrossMHA-Features-Permute_{i+1}')(features)
            x, _ = vcm.ExpandedQueriesMHA(output_dim=embs_dim, value_dim=embs_dim, key_dim=embs_dim, num_heads=num_heads, name=f'CrossMHA_{i+1}')(V, Q, V, Q_pos_enc)
            x = tf.keras.layers.Permute([2,1,3], name=f'PreCrossMHA-Features-UnPermute_{i+1}')(x)
            features = vcm.AddNorm(norm_axis=-1, name=f'PostCrossMHA-AddNorm_{i+1}')([features, x])

        if sample_queries_num is not None:
            query_points = vcm.QuerySamplingLayer(queries_num=sample_queries_num, mid_layers=2, mid_units=embs_dim, activation='relu', dropout=0.0, name=f'Query-Sample_{i+1}')([sample_features, sample_inputs])
            query_pos_enc = self_enc_func(emb_dim=embs_dim//num_heads, height=size, expand_b=False, name=f'Query{enc_label}RadialEncoding_{i+1}')(sample_inputs, query_points)
            query_samples = vcm.SampleQueryExtractionLayer(cut_off=1, gamma=2, name=f'Query-Features_{i+1}')([features, query_points])
            sample_features = vcm.SampleQueryMessagePassing(mid_layers=3, mid_units=embs_dim*2, activation='relu', dropout=0.0, name=f'Query-Message_{i+1}')(sample_features, query_samples, query_pos_enc)


        x = FFN(mid_layers=out_mid_layers, mid_units=embs_dim*2, output_units=embs_dim, dropout=dropout, activation=activation, name=f'vdDecoder-FFN_{i+1}')(sample_features)
        sample_features = vcm.AddNorm(norm_axis=-1, name=f'vdPostFFN-AddNorm_{i+1}')([sample_features, x])

        if features_update & (i<attns_num-1):
            u = vcm.QuerySamplesFeaturesMHAUpdate(name=f'Features-MHA-update_{i}')([sample_features, scores])
            features = vcm.AddNorm(norm_axis=-1, name=f'Features-PostMHA-AddNorm_{i}')([features, u])

            u = FFN(mid_layers=out_mid_layers, mid_units=embs_dim*2, output_units=embs_dim, dropout=dropout, activation=activation, name=f'Features-Decoder-FFN_{i}')(features)
            features = vcm.AddNorm(norm_axis=-1, name=f'Features-PostFFN-AddNorm_{i}')([features, u])
        
    vec_preds_col = vcm.SampleRadialSearchHead(num_samples=sample_points, 
                                    ffn_mid_layers=out_mid_layers, 
                                    mid_units=embs_dim*2, 
                                    activation=activation, 
                                    dropout=0.0, 
                                    angle_pred=angle_based_preds, 
                                    thickness_pred=thickness_pred, 
                                    name='SRShead')(sample_features, sample_inputs)
    
    vecs = tf.keras.layers.Identity(name='vecs')(vec_preds_col[0])
    class_preds = tf.keras.layers.Identity(name='class')(vec_preds_col[1])
    vec_preds_output = {'vecs': vecs, 'class': class_preds}

    if thickness_pred:
        thickness_preds = tf.keras.layers.Identity(name='thickness')(vec_preds_col[2])
        vec_preds_output['thickness'] = thickness_preds

    model = tf.keras.Model(inputs={'img': img_inputs, 'sample_points': sample_inputs}, 
                           outputs=vec_preds_output, 
                           name=name)
    
    return model