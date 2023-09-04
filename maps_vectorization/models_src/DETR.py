import tensorflow as tf

class SinePositionEncoding(tf.keras.layers.Layer):
    def __init__(self, fixed_dims=False, temperature=10000, **kwargs):
        super(SinePositionEncoding, self).__init__(**kwargs)

        self.fixed_dims = fixed_dims
        self.temperature = temperature

        self.fixed_dims = fixed_dims

    def build(self, input_shape):
        num_pos_features = input_shape[-1] // (1 if self.fixed_dims else 2)
        mask = tf.ones((1,)+input_shape[1:-1], dtype=tf.float32)
        y_embed = tf.math.cumsum(mask, axis=1)
        x_embed = tf.math.cumsum(mask, axis=2)

        dim_t = tf.range(num_pos_features, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_features)

        pos_x = x_embed[..., tf.newaxis] / dim_t
        pos_y = y_embed[..., tf.newaxis] / dim_t

        pos_x = tf.stack([tf.math.sin(pos_x[..., 0::2]),
                          tf.math.cos(pos_x[..., 1::2])], axis=4)

        pos_y = tf.stack([tf.math.sin(pos_y[..., 0::2]),
                          tf.math.cos(pos_y[..., 1::2])], axis=4)

        shape = [tf.shape(pos_x)[i] for i in range(3)] + [-1]
        pos_x = tf.reshape(pos_x, shape)
        pos_y = tf.reshape(pos_y, shape)

        if self.fixed_dims:
            self.pos_emb = (pos_x + pos_y)/2
        else:
            self.pos_emb = tf.concat([pos_y, pos_x], axis=3)

        #self.output_reshape = (tf.reduce_prod(input_shape[1:-1]), input_shape[-1])
        self.pos_emb = tf.reshape(self.pos_emb, (tf.reduce_prod(input_shape[1:-1]), input_shape[-1]))

    def call(self, inputs):        
        return self.pos_emb
    
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1), **kwargs):
        super(LearnablePositionalEncoding, self).__init__(**kwargs)

        self.initializer = initializer

    def build(self, input_shape):
        self.H,self.W,self.D = input_shape[1], input_shape[2], input_shape[3]

        self.rows_embed = self.add_weight(shape=(self.H,1,self.D//2), initializer=self.initializer, trainable=True)
        self.cols_embed = self.add_weight(shape=(1,self.W,self.D//2), initializer=self.initializer, trainable=True)

    def call(self, inputs):

        pos = tf.concat([
            tf.repeat(self.cols_embed, self.H, axis=0),
            tf.repeat(self.rows_embed, self.W, axis=1)
        ], axis=-1)

        pos = tf.expand_dims(tf.reshape(pos, (self.H*self.W, self.D)), axis=0)

        return pos
    
class FlatLearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, initializer=tf.keras.initializers.GlorotUniform(), **kwargs):
        super(FlatLearnablePositionalEncoding, self).__init__(**kwargs)

        self.initializer = initializer

    def build(self, input_shape):
        seq_len, dims = input_shape[1], input_shape[2]

        self.pos = self.add_weight(shape=(seq_len, dims), initializer=self.initializer, trainable=True)

    def call(self, inputs):
        return self.pos

class FFN(tf.keras.layers.Layer):
    def __init__(self, mid_layers=1, 
                 mid_units=1024, 
                 output_units=512, 
                 dropout=0.0, 
                 activation='relu',
                 **kwargs):
        super(FFN, self).__init__(**kwargs)

        self.seq = tf.keras.Sequential(
            [tf.keras.layers.Dense(mid_units, activation=activation) for _ in range(mid_layers)] + \
            [tf.keras.layers.Dropout(dropout)] + \
            [tf.keras.layers.Dense(output_units)]
        )

    def call(self, inputs, training=None):
        return self.seq(inputs, training=training)
    
class HeadsPermuter(tf.keras.layers.Layer):
    def __init__(self, num_heads, emb_dim, reverse=False,**kwargs):
        super(HeadsPermuter, self).__init__(**kwargs)

        permute = tf.keras.layers.Permute((2,1,3))
        direction = -1 if reverse else 1
        target_shape = (-1, num_heads*emb_dim) if reverse else (-1,num_heads, emb_dim)

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Reshape(target_shape),
            permute
        ][::direction])

    def call(self, inputs):
        return self.seq(inputs)

    
class MHA(tf.keras.layers.Layer):
    def __init__(self,
                 output_dim,
                 value_dim,
                 key_dim,
                 num_heads,
                 **kwargs):
        super(MHA, self).__init__(**kwargs)

        self.Q_d = tf.keras.layers.Dense(key_dim)
        self.K_d = tf.keras.layers.Dense(key_dim)
        self.V_d = tf.keras.layers.Dense(value_dim)
        self.O_d = tf.keras.layers.Dense(output_dim)

        self.softmax = tf.keras.activations.softmax
        self.denominator = tf.math.sqrt(tf.cast(key_dim, tf.float32))

        self.QK_head_extractior = HeadsPermuter(num_heads, key_dim//num_heads, reverse=False)
        self.V_head_extractior = HeadsPermuter(num_heads, value_dim//num_heads, reverse=False)
        self.output_perm = HeadsPermuter(num_heads, value_dim//num_heads, reverse=True)

    def call(self, V, Q, K):
        Q = self.QK_head_extractior(self.Q_d(Q))
        K = self.QK_head_extractior(self.K_d(K))

        scores = tf.matmul(Q, K, transpose_b=True)/self.denominator
        weights = self.softmax(scores, axis=-1)

        V = self.V_head_extractior(self.V_d(V))

        return self.O_d(self.output_perm(tf.matmul(weights, V)))



class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 attn_dim=512, 
                 key_dim=512, 
                 num_heads=4, 
                 dropout=0.0,
                 FFN_mid_layers=1, 
                 FFN_mid_units=2048,
                 FFN_activation='relu',
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.attn_dropout, self.output_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(2)]
        self.attn_addnorm, self.output_addnorm = [tf.keras.Sequential([
            tf.keras.layers.Add(),
            tf.keras.layers.LayerNormalization()])
            for _ in range(2)]

        self.FFN = FFN(FFN_mid_layers, FFN_mid_units, attn_dim, dropout, FFN_activation)

        self.MHA = MHA(attn_dim, attn_dim, key_dim, num_heads)

    def call(self, V, pos_enc, training=None):
        Q = K = V + pos_enc

        # Multi-Head-Attention
        V = self.attn_addnorm([V, self.attn_dropout(self.MHA(V, Q, K), training=training)])

        # Feed-Forward-Network
        V = self.output_addnorm([V, self.output_dropout(self.FFN(V), training=training)])

        return V
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 attn_dim=512, 
                 key_dim=512, 
                 num_heads=4, 
                 dropout=0.0,
                 FFN_mid_layers=1, 
                 FFN_mid_units=2048,
                 FFN_activation='relu',
                 **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.QO_dropout, self.MEM_dropout, self.output_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(3)]
        self.QO_addnorm, self.MEM_addnorm, self.output_addnorm = [tf.keras.Sequential([
            tf.keras.layers.Add(),
            tf.keras.layers.LayerNormalization()])
            for _ in range(3)]

        self.FFN = FFN(FFN_mid_layers, FFN_mid_units, attn_dim, dropout, FFN_activation)

        self.QO_MHA = MHA(attn_dim, attn_dim, key_dim, num_heads)
        self.MEM_MHA = MHA(attn_dim, attn_dim, key_dim, num_heads)

    def call(self, QO, memory, pos_enc, QO_enc, training=None):

        QO_Q = QO_K = QO + QO_enc
        
        QO = self.QO_addnorm([QO,self.QO_dropout(self.QO_MHA(QO, QO_Q, QO_K), training=training)])

        K = memory+pos_enc
        Q = QO + QO_enc
        QO = self.MEM_addnorm([QO,self.MEM_dropout(self.MEM_MHA(memory, Q, K), training=training)])

        QO = self.output_addnorm([QO, self.output_dropout(self.FFN(QO), training=training)])

        return QO
    
class DETRUpConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), strides=(2,2), mid_convs=2, dropout=0.0, activation='relu', **kwargs):
        super(DETRUpConv, self).__init__(**kwargs)

        self.upconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, activation=activation, padding='same')
        self.convs = [tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation) for n in range(mid_convs)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        x = self.upconv(inputs)
        for conv in self.convs:
            x = conv(x)
        
        x = self.dropout(x, training=training)

        return x
    

class DETRTransformer(tf.keras.Model):
    def __init__(self, 
                 input_shape=(256,256,3),
                 output_dim=100,
                 attn_dim=512, 
                 key_dim=512, 
                 num_heads=4, 
                 dropout=0.0,
                 FFN_mid_layers=1, 
                 FFN_mid_units=2048,
                 FFN_activation='relu',
                 layers_num = 6,
                 pos_encoder = SinePositionEncoding(name='PositionalEncoder'),
                 query_objects_encoder = FlatLearnablePositionalEncoding(name='QueryObjects-PosEncoder'),
                 upconv_blocks = 4,
                 **kwargs):
        super(DETRTransformer, self).__init__(**kwargs)

        backbone = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=input_shape)
        self.backbone = tf.keras.Model(backbone.inputs, backbone.layers[-2].output, name='Backbone')
        self.backbone.trainable = False
        backbone_shape = self.backbone.output_shape

        self.feature_extraction = tf.keras.layers.Conv2D(attn_dim, 1, name='Features-Extraction')

        self.image_flatten = tf.keras.layers.Reshape((-1, attn_dim), name='Image-Flattening')

        self.pos_enc_func = pos_encoder
        self.QO_enc_func = query_objects_encoder

        self.encoder_layers = [EncoderLayer(attn_dim, key_dim, num_heads, dropout, 
                                            FFN_mid_layers, FFN_mid_units, FFN_activation,
                                            name=f'Encoder-{n}') for n in range(layers_num)]
        
        self.decoder_layers = [DecoderLayer(attn_dim, key_dim, num_heads, dropout, 
                                            FFN_mid_layers, FFN_mid_units, FFN_activation,
                                            name=f'Decoder-{n}') for n in range(layers_num)]
        
        self.img_unsqueeze = tf.keras.layers.Reshape((backbone_shape[1], backbone_shape[2], attn_dim), name='IMG-Unsqueeze')

        self.upconvs = [DETRUpConv(filters=attn_dim, name=f'UpConv-{n}') for n in range(upconv_blocks)]
        self.output_conv = tf.keras.layers.Conv2D(output_dim, 1, activation='sigmoid',name='Output-Conv')

        self.call(tf.keras.layers.Input(input_shape))
        

    def call(self, inputs, training=None):

        #backbone
        features = self.feature_extraction(self.backbone(inputs, training=training))
        
        pos_enc = self.pos_enc_func(features)
        memory = self.image_flatten(features)

        #Encoder
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory, pos_enc, training=training)

        #Decoder
        QO = tf.zeros_like(memory)
        QO_enc = self.QO_enc_func(QO)

        for decoder_layer in self.decoder_layers:
            QO = decoder_layer(QO, memory, pos_enc, QO_enc, training=training)

        # Upscale output images
        output = self.img_unsqueeze(QO)

        for upconv in self.upconvs:
            output = upconv(output)

        output = self.output_conv(output)
        return tf.clip_by_value(output, 1e-8, 1-1e-8)