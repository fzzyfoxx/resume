import tensorflow as tf
import math
from models_src.CombinedMetricsModel import CombinedMetricsModel

class SinePositionEncoding(tf.keras.layers.Layer):
    """Sine-based positional encoding layer.

    Produces fixed or concatenated sine/cosine positional embeddings matching
    the spatial shape of a feature map.
    """
    def __init__(self, fixed_dims=False, temperature=10000, normalize=False, **kwargs):
        """Initialize the positional encoding.

        Args:
            fixed_dims (bool): If True, use same number of dims for x and y.
            temperature (int): Temperature scaling factor for frequency bands.
            normalize (bool): If True, normalize coordinates to [0, 2*pi].
        """
        super(SinePositionEncoding, self).__init__(**kwargs)

        self.fixed_dims = fixed_dims
        self.temperature = temperature
        self.normalize = normalize

    def build(self, input_shape):
        """Build the positional embeddings using the provided input shape.

        Args:
            input_shape (TensorShape): Shape of the feature map input; used to
                compute positional embedding dimensions.
        """
        num_pos_features = input_shape[-1] // (1 if self.fixed_dims else 2)
        mask = tf.ones((1,)+input_shape[-3:-1], dtype=tf.float32)
        y_embed = tf.math.cumsum(mask, axis=1)
        x_embed = tf.math.cumsum(mask, axis=2)

        if self.normalize:
            y_embed = y_embed/input_shape[-3]*math.pi*2
            x_embed = x_embed/input_shape[-2]*math.pi*2

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

        self.pos_emb = tf.reshape(self.pos_emb, (tf.reduce_prod(input_shape[-3:-1]), input_shape[-1]))

    def call(self, inputs=None):        
        """Return the computed positional embedding tensor.

        Args:
            inputs: Ignored; kept for Keras compatibility.

        Returns:
            Tensor of shape (H*W, D) containing positional embeddings.
        """
        return self.pos_emb
    
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    """Learnable 2D positional encoding layer.

    Creates row and column embeddings and concatenates them to form a
    positional embedding for each spatial location.
    """
    def __init__(self, initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1), **kwargs):
        """Initialize the learnable positional encoding.

        Args:
            initializer: Initializer for the learnable embedding weights.
        """
        super(LearnablePositionalEncoding, self).__init__(**kwargs)

        self.initializer = initializer

    def build(self, input_shape):
        """Create and combine row/column embedding weights.

        Args:
            input_shape: Shape of the input tensor used to infer H, W, D.
        """
        self.H,self.W,self.D = input_shape[1], input_shape[2], input_shape[3]

        self.rows_embed = self.add_weight(shape=(self.H,1,self.D//2), initializer=self.initializer, trainable=True)
        self.cols_embed = self.add_weight(shape=(1,self.W,self.D//2), initializer=self.initializer, trainable=True)

        pos = tf.concat([
            tf.repeat(self.cols_embed, self.H, axis=0),
            tf.repeat(self.rows_embed, self.W, axis=1)
        ], axis=-1)

        self.pos = tf.reshape(pos, (self.H*self.W, self.D))

    def call(self, inputs):
        """Return the flattened positional embeddings.

        Args:
            inputs: Ignored; kept to match Keras layer signature.

        Returns:
            Tensor of shape (H*W, D) containing learned positional embeddings.
        """
        return self.pos
    
class FlatLearnablePositionalEncoding(tf.keras.layers.Layer):
    """Flat learnable positional encoding for query embeddings.

    Maintains a small trainable matrix of shape (1, num_queries, emb_dim).
    """
    def __init__(self, num_queries, emb_dim, initializer=tf.keras.initializers.GlorotUniform(), **kwargs):
        """Initialize the flat learnable positional encoding.

        Args:
            num_queries (int): Number of query positions.
            emb_dim (int): Embedding dimensionality for each query.
            initializer: Weight initializer for the positional embeddings.
        """
        super(FlatLearnablePositionalEncoding, self).__init__(**kwargs)

        self.initializer = initializer
        self.num_queries = num_queries
        self.emb_dim = emb_dim

        self.pos = self.add_weight(shape=(1,self.num_queries, self.emb_dim), initializer=self.initializer, trainable=True)

    def call(self, inputs=None):
        """Return the learnable query positional embeddings.

        Args:
            inputs: Ignored; kept for compatibility.

        Returns:
            Tensor of shape (1, num_queries, emb_dim).
        """
        return self.pos
    
    def compute_output_shape(self, input_shape):
        """Return the static output shape of this layer.

        Args:
            input_shape: Input shape passed by Keras.
        """
        return (1,self.num_queries, self.emb_dim)

class FFN(tf.keras.layers.Layer):
    """Feed-Forward Network (MLP) used inside transformer blocks.

    Composed of several Dense layers, optional dropout, and a final Dense
    projection to the desired output dimension.
    """
    def __init__(self, mid_layers=1, 
                 mid_units=1024, 
                 output_units=512, 
                 dropout=0.0, 
                 activation='relu',
                 kernel_initializer='he_normal',
                 **kwargs):
        """Initialize the FFN.

        Args:
            mid_layers (int): Number of intermediate Dense layers.
            mid_units (int): Units in each intermediate Dense layer.
            output_units (int): Units in the output Dense layer.
            dropout (float): Dropout rate applied between layers.
            activation (str): Activation for intermediate layers.
            kernel_initializer: Kernel initializer for Dense layers.
        """
        super(FFN, self).__init__(**kwargs)

        self.seq = tf.keras.Sequential(
            [tf.keras.layers.Dense(mid_units, activation=activation, kernel_initializer=kernel_initializer) for _ in range(mid_layers)] + \
            [tf.keras.layers.Dropout(dropout)] + \
            [tf.keras.layers.Dense(output_units, kernel_initializer=kernel_initializer)]
        )

    def call(self, inputs, training=None):
        """Run the FFN on inputs.

        Args:
            inputs: Input tensor.
            training (bool): Whether the layer is in training mode.

        Returns:
            Transformed tensor after MLP and dropout.
        """
        return self.seq(inputs, training=training)
    
    def compute_output_shape(self, input_shape):
        """Return the output shape produced by the internal Sequential model.

        Args:
            input_shape: Shape of the input tensor.
        """
        return self.seq.compute_output_shape(input_shape)
        
class HeadsPermuter(tf.keras.layers.Layer):
    """Permute tensor shapes to split or combine attention heads.

    This layer reshapes and transposes tensors to expose a heads dimension
    or to collapse it back into the embedding dimension.
    """
    def __init__(self, num_heads, reverse=False,**kwargs):
        """Initialize the permuter.

        Args:
            num_heads (int): Number of attention heads to expose/collapse.
            reverse (bool): If True, perform reverse permutation (collapse heads).
        """
        super(HeadsPermuter, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.reverse = reverse

    def build(self, input_shape):
        """Prepare shape metadata used for permutations.

        Args:
            input_shape: Shape of the tensor that will be permuted.
        """
        dynamic_dims = 3 if self.reverse else 2
        static_dims = len(input_shape)-dynamic_dims
        self.extra_dims_shape = input_shape[1:-dynamic_dims]

        self.length = input_shape[-2]
        self.emb_dim = input_shape[-1] if self.reverse else input_shape[-1]//self.num_heads

        self.perm = list(range(static_dims))+[d+static_dims for d in [1,0,2]]

    def _permutation(self, x):
        """Reshape and transpose to expose a head dimension.

        Args:
            x: Input tensor to permute.

        Returns:
            Tensor with shape (..., length, num_heads, emb_dim).
        """
        x = tf.reshape(x, (-1,)+self.extra_dims_shape+(self.length, self.num_heads, self.emb_dim))
        x = tf.transpose(x, perm=self.perm)
        return x

    def _reverse_permutation(self, x):
        """Transpose and reshape to collapse heads into the embedding dim.

        Args:
            x: Input tensor with explicit head dimension.

        Returns:
            Tensor with heads collapsed back to the embedding dimension.
        """
        x = tf.transpose(x, perm=self.perm)
        x = tf.reshape(x, (-1,)+self.extra_dims_shape+(self.length, self.num_heads*self.emb_dim))
        return x

    def call(self, inputs):
        """Dispatch to permutation or reverse permutation based on config.

        Args:
            inputs: Tensor to permute.

        Returns:
            Permuted tensor.
        """
        if self.reverse:
            return self._reverse_permutation(inputs)
        return self._permutation(inputs)

class MHA(tf.keras.layers.Layer):
    """Multi-Head Attention wrapper with optional masking and weight transpose.

    Implements linear projections for Q, K, V, per-head splitting and
    recombination followed by a final output projection.
    """
    def __init__(self,
                 output_dim,
                 value_dim,
                 key_dim,
                 num_heads,
                 transpose_weights=False,
                 softmax_axis=-1,
                 soft_mask=False,
                 return_weights=False,
                 **kwargs):
        """Create the MHA layer and its internal linear projections.

        Args:
            output_dim (int): Output dimensionality after combining heads.
            value_dim (int): Dimensionality of value projections.
            key_dim (int): Dimensionality of key/query projections.
            num_heads (int): Number of attention heads.
            transpose_weights (bool): If True, use transposed weight multiplication.
            softmax_axis (int): Axis for the attention softmax.
            soft_mask (bool): If True, add mask logits instead of hard masking.
            return_weights (bool): If True, return (output, attention_weights).
        """
        super(MHA, self).__init__(**kwargs)

        self.Q_d = tf.keras.layers.Dense(key_dim, name='Q_Dense')
        self.K_d = tf.keras.layers.Dense(key_dim, name='K_Dense')
        self.V_d = tf.keras.layers.Dense(value_dim, name='V_Dense')
        self.O_d = tf.keras.layers.Dense(output_dim, name='Out_Dense')

        self.softmax = tf.keras.activations.softmax
        self.denominator = tf.math.sqrt(tf.cast(key_dim, tf.float32))

        self.Q_head_extractor = HeadsPermuter(num_heads, reverse=False)
        self.K_head_extractor = HeadsPermuter(num_heads, reverse=False)
        self.V_head_extractor = HeadsPermuter(num_heads, reverse=False)
        self.output_perm = HeadsPermuter(num_heads, reverse=True)

        self.T = transpose_weights
        self.softmax_axis = softmax_axis

        self.soft_mask = soft_mask
        self.return_weights = return_weights

    def call(self, V, Q, K, mask=None):
        """Compute attention output (and optionally weights).

        Args:
            V: Value tensor.
            Q: Query tensor.
            K: Key tensor.
            mask: Optional mask tensor broadcastable to attention scores.

        Returns:
            If return_weights is False: attention output tensor.
            If return_weights is True: tuple(output, attention_weights).
        """
        Q = self.Q_head_extractor(self.Q_d(Q))
        K = self.K_head_extractor(self.K_d(K))

        scores = tf.matmul(Q, K, transpose_b=True)/self.denominator
        if mask is not None:
            if self.soft_mask:
                scores = scores + mask
            else:
                cross_mask = tf.expand_dims(tf.matmul(mask, mask, transpose_b=True), axis=1)
                scores = scores+((cross_mask-1)*math.inf)
        weights = self.softmax(scores, axis=self.softmax_axis)

        V = self.V_head_extractor(self.V_d(V))
        if not self.T:
            V = tf.matmul(weights, V)
        else:
            V *= tf.transpose(weights, perm=[0,1,3,2])

        V = self.O_d(self.output_perm(V))

        if (mask is not None) & (self.soft_mask==False):
            return V*mask
        
        if self.return_weights:
            return V, weights
        return V

class DeepLayerNormalization(tf.keras.layers.Layer):
    """Layer normalization variant that normalizes across a specified axis.

    Computes mean/std across the provided axis and returns a normalized tensor.
    """
    def __init__(self, norm_axis=1, **kwargs):
        """Initialize with the axis to normalize over.

        Args:
            norm_axis (int): Axis to compute mean and standard deviation over.
        """
        super(DeepLayerNormalization, self).__init__(**kwargs)
        self.axis = norm_axis

    def call(self, inputs, training=None):
        """Normalize inputs by mean and standard deviation along `axis`.

        Args:
            inputs: Input tensor to normalize.
            training: Ignored; kept for API compatibility.

        Returns:
            Normalized tensor with same shape as inputs.
        """
        norm_mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        norm_std = tf.math.reduce_std(inputs, axis=self.axis, keepdims=True)
        return tf.math.divide_no_nan((inputs-norm_mean),norm_std)

class EncoderLayer(tf.keras.layers.Layer):
    """Single transformer encoder layer with attention and FFN.

    Combines multi-head attention with residual/normalization and a
    feed-forward network.
    """
    def __init__(self, 
                 attn_dim=512, 
                 key_dim=512, 
                 num_heads=4, 
                 dropout=0.0,
                 FFN_mid_layers=1, 
                 FFN_mid_units=2048,
                 FFN_activation='relu',
                 deep_normalization=True,
                 soft_mask=False,
                 **kwargs):
        """Initialize encoder sublayers and normalization modules.

        Args:
            attn_dim, key_dim, num_heads, dropout, FFN_*: Standard transformer
                hyperparameters used to instantiate sublayers.
            deep_normalization (bool): If True use DeepLayerNormalization.
            soft_mask (bool): If True use additive soft masking in attention.
        """
        super(EncoderLayer, self).__init__(**kwargs)

        self.attn_dropout, self.output_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(2)]
        self.attn_addnorm, self.output_addnorm = [tf.keras.Sequential([
            tf.keras.layers.Add(),
            (DeepLayerNormalization(norm_axis=1) if deep_normalization else tf.keras.layers.LayerNormalization())])
            for _ in range(2)]

        self.FFN = FFN(FFN_mid_layers, FFN_mid_units, attn_dim, dropout, FFN_activation)

        self.MHA = MHA(attn_dim, attn_dim, key_dim, num_heads, soft_mask)

    def call(self, V, pos_enc=None, Q=None, K=None, training=None):
        """Run one encoder layer pass.

        Args:
            V: Input tensor (memory) of shape (batch, length, dim).
            pos_enc: Optional positional encodings added to Q/K.
            Q, K: Optional explicit query/key inputs; if None, V is used.
            training: Passed to dropout layers.

        Returns:
            Updated memory tensor after attention and FFN.
        """
        if Q is None:
            Q = V

        if K is None:
            K = V

        if pos_enc is not None:
            Q = Q + pos_enc
            K = K + pos_enc

        # Multi-Head-Attention
        V = self.attn_addnorm([V, self.attn_dropout(self.MHA(V, Q, K), training=training)])

        # Feed-Forward-Network
        V = self.output_addnorm([V, self.output_dropout(self.FFN(V), training=training)])

        return V
    
class DecoderLayer(tf.keras.layers.Layer):
    """Single transformer decoder layer with self- and cross-attention.

    Accepts query object embeddings and attends to encoder memory with
    positional encodings.
    """
    def __init__(self, 
                 attn_dim=512, 
                 key_dim=512, 
                 num_heads=4, 
                 dropout=0.0,
                 FFN_mid_layers=1, 
                 FFN_mid_units=2048,
                 FFN_activation='relu',
                 **kwargs):
        """Initialize decoder sublayers used for cross-attention.

        Args:
            attn_dim, key_dim, num_heads, dropout, FFN_*: Standard transformer
                hyperparameters used to instantiate sublayers.
        """
        super(DecoderLayer, self).__init__(**kwargs)

        self.QO_dropout, self.MEM_dropout, self.output_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(3)]
        self.QO_addnorm, self.MEM_addnorm, self.output_addnorm = [tf.keras.Sequential([
            tf.keras.layers.Add(),
            tf.keras.LayerNormalization(axis=-1)])
            for _ in range(3)]

        self.FFN = FFN(FFN_mid_layers, FFN_mid_units, attn_dim, dropout, FFN_activation)

        self.QO_MHA = MHA(attn_dim, attn_dim, key_dim, num_heads)
        self.MEM_MHA = MHA(attn_dim, attn_dim, key_dim, num_heads)

    def call(self, QO, memory, pos_enc, QO_enc, training=None):
        """Run one decoder layer pass.

        Args:
            QO: Query object embeddings to update.
            memory: Encoder memory tensor to attend to.
            pos_enc: Positional encodings applied to memory.
            QO_enc: Positional encodings for the queries.
            training: Passed to dropout layers.

        Returns:
            Updated query object tensor after self- and cross-attention.
        """
        QO_Q = QO_K = QO + QO_enc
        
        QO = self.QO_addnorm([QO,self.QO_dropout(self.QO_MHA(QO, QO_Q, QO_K), training=training)])

        K = memory+pos_enc
        Q = QO + QO_enc
        QO = self.MEM_addnorm([QO,self.MEM_dropout(self.MEM_MHA(memory, Q, K), training=training)])

        QO = self.output_addnorm([QO, self.output_dropout(self.FFN(QO), training=training)])

        return QO
    
class DETRUpConv(tf.keras.layers.Layer):
    """Simple upsampling block composed of Conv2DTranspose and Conv2D layers.

    Used to progressively upscale low-resolution feature maps to image size.
    """
    def __init__(self, filters, kernel_size=(3,3), strides=(2,2), mid_convs=2, dropout=0.0, activation='relu', **kwargs):
        """Initialize up-convolutional block.

        Args:
            filters: Number of filters for conv layers.
            kernel_size: Kernel size for conv operations.
            strides: Strides used for Conv2DTranspose upsampling.
            mid_convs: Number of intermediate Conv2D layers.
            dropout: Dropout rate after convolutions.
            activation: Activation function for conv layers.
        """
        super(DETRUpConv, self).__init__(**kwargs)

        self.upconv = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, activation=activation, padding='same')
        self.convs = [tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=activation) for n in range(mid_convs)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        """Apply upsampling and convolution operations.

        Args:
            inputs: Input feature map tensor.
            training: Passed to dropout layer.

        Returns:
            Upsampled feature map tensor.
        """
        x = self.upconv(inputs)
        for conv in self.convs:
            x = conv(x)
        
        x = self.dropout(x, training=training)

        return x

class ZeroLikeLayer(tf.keras.layers.Layer):
    """Layer that returns a zero tensor matching the input's shape.

    Primarily used to create zero-initialized query object embeddings.
    """
    def __init__(self, **kwargs):
        """Initialize the zero-like layer (no parameters).
        """
        super(ZeroLikeLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        """Return a zero tensor with the same shape and dtype as `inputs`.

        Args:
            inputs: Input tensor whose shape/dtype will be matched.
            training: Ignored; present for API compatibility.

        Returns:
            Tensor of zeros with the same shape as `inputs`.
        """
        return tf.zeros_like(inputs)

class DETRTransformer(CombinedMetricsModel):
    """DETR-like transformer model combining a CNN backbone with encoder/decoder.

    Builds a complete model: backbone feature extractor, transformer encoder
    stack, decoder stack over learnable query embeddings, and either a mask
    head or detection head producing class confidences and bounding boxes.
    """
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
                 upconv_blocks = 4,
                 queries_num=50,
                 FFN_out_layers=3,
                 FFN_out_mid_units=1024,
                 FFN_out_activation='relu',
                 mask_output=False,
                 metrics=[],
                 **kwargs):
        """Construct the DETRTransformer model.

        Args:
            input_shape: Input image shape (H,W,C).
            output_dim: Number of channels in mask output or detection head dim.
            attn_dim, key_dim, num_heads, dropout, FFN_*: Transformer hyperparams.
            layers_num: Number of encoder/decoder layers.
            pos_encoder: Positional encoder layer instance.
            upconv_blocks: Number of upsampling blocks when producing masks.
            queries_num: Number of learnable query embeddings for detection.
            mask_output: Whether to build a per-pixel mask head instead of detector.
            metrics: List of metrics to register on the Keras model.
        """
        backbone = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=input_shape)
        backbone = tf.keras.Model(backbone.inputs, backbone.layers[-2].output, name='Backbone')
        backbone_shape = backbone.output_shape

        feature_extraction = tf.keras.layers.Conv2D(attn_dim, 1, name='Features-Extraction')

        image_flatten = tf.keras.layers.Reshape((-1, attn_dim), name='Image-Flattening')

        pos_enc_func = pos_encoder
        QO_enc_func = FlatLearnablePositionalEncoding(queries_num, key_dim, name='QueryObjects-PosEncoder')

        encoder_layers = [EncoderLayer(attn_dim, key_dim, num_heads, dropout, 
                                            FFN_mid_layers, FFN_mid_units, FFN_activation,
                                            name=f'Encoder-{n}') for n in range(layers_num)]
        
        decoder_layers = [DecoderLayer(attn_dim, key_dim, num_heads, dropout, 
                                            FFN_mid_layers, FFN_mid_units, FFN_activation,
                                            name=f'Decoder-{n}') for n in range(layers_num)]
        mask_output = mask_output
        if mask_output:
            img_unsqueeze = tf.keras.layers.Reshape((backbone_shape[1], backbone_shape[2], attn_dim), name='IMG-Unsqueeze')
            upconvs = [DETRUpConv(filters=attn_dim, name=f'UpConv-{n}') for n in range(upconv_blocks)]
            output_conv = tf.keras.layers.Conv2D(output_dim, 1, activation='sigmoid',name='Output-Conv')
        else:
            out_FFN = FFN(FFN_out_layers, FFN_out_mid_units, 5, dropout, FFN_out_activation, name='Out-FFN')
            out_sigmoid = tf.keras.layers.Activation('sigmoid', name='Out-Sigmoid')

        inputs = tf.keras.layers.Input(input_shape)
        #backbone
        features = feature_extraction(backbone(inputs))
        
        pos_enc = pos_enc_func(features)
        memory = image_flatten(features)

        #Encoder
        for encoder_layer in encoder_layers:
            memory = encoder_layer(memory, pos_enc)

        #Decoder
        QO_enc = QO_enc_func(features)
        QO = ZeroLikeLayer(name='QueryObjects')(QO_enc)

        for decoder_layer in decoder_layers:
            QO = decoder_layer(QO, memory, pos_enc, QO_enc)

        if mask_output:
            # Upscale output images
            output = img_unsqueeze(QO)

            for upconv in upconvs:
                output = upconv(output)

            output = output_conv(output)
            output = tf.clip_by_value(output, 1e-8, 1-1e-8)
        else:
            output = out_sigmoid(out_FFN(QO))
            confidence, bboxes = self._bbox_decoding(output)
            confidence = tf.squeeze(confidence, axis=2)
            output = {'class': confidence, 'bbox': bboxes}

        super(DETRTransformer, self).__init__(inputs=inputs, outputs=output, **kwargs)

        self.add_metrics(metrics)

    def _bbox_decoding(self, x):
        """Decode network outputs into confidence and bounding boxes.

        Args:
            x: Raw network output tensor with bbox param encoding.

        Returns:
            Tuple (confidence, bboxes) where confidence has shape (...,1) and
            bboxes are clipped to [0,1] coordinates as (ymin,xmin,ymax,xmax).
        """
        C, YX, HW = x[...,:1], x[...,1:3], x[...,3:]*0.5

        bboxes = tf.concat([tf.clip_by_value(YX-HW, [0,0], [1,1]), tf.clip_by_value(YX+HW, [0,0], [1,1])], axis=-1)
        return C, bboxes