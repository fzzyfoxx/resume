import tensorflow as tf
import math
import matplotlib.pyplot as plt
from models_src.DETR import FFN, MHA, SinePositionEncoding


class SineColorEmbeddigs(tf.keras.layers.Layer):
    def __init__(self, ratios=[0.5,1,2,4,8,16], **kwargs):
        super(SineColorEmbeddigs, self).__init__(**kwargs)

        self.ratios = tf.constant(ratios, tf.float32)*math.pi

    def build(self, input_shape):

        self.reshape = tf.keras.layers.Reshape(input_shape[1:-1]+[input_shape[-1]*tf.shape(self.ratios)[0]])

        for _ in input_shape:
            self.ratios = self.ratios[tf.newaxis]

    def call(self, inputs):
        return self.reshape(tf.math.sin(tf.expand_dims(inputs, axis=-1)*self.ratios))
    

class LinearPositionalEmbeddings(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, concat=True, **kwargs):
        super(LinearPositionalEmbeddings, self).__init__(**kwargs)

        self.scale = scale
        self.concat = concat

    def build(self, input_shape):

        H, W = input_shape[-3], input_shape[-2]
        h = tf.expand_dims(tf.repeat((tf.range(H)/H)[:,tf.newaxis], W, axis=1), axis=-1)
        w = tf.expand_dims(tf.repeat((tf.range(W)/W)[tf.newaxis], H, axis=0), axis=-1)

        emb = tf.concat([h,w], axis=-1)
        for d in input_shape[1:-3]:
            emb = emb[tf.newaxis]
            emb = tf.repeat(emb, d, axis=0)

        self.emb = tf.cast(emb[tf.newaxis], tf.float32)*self.scale

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        emb = tf.repeat(self.emb, B, axis=0)
        if self.concat:
            return tf.concat([inputs, emb], axis=-1)
        return emb
    
def LinearPositions(shape=(32,32), flat_output=False, normalize=False, invert_xy=False):
    H, W = shape[0], shape[1]
    h = tf.repeat((tf.range(H))[:,tf.newaxis], W, axis=1)
    w = tf.repeat((tf.range(W))[tf.newaxis], H, axis=0)

    emb = tf.stack([h,w], axis=-1)
    if normalize:
        emb = emb/(tf.constant(shape)[tf.newaxis, tf.newaxis]-1)
    if flat_output:
        emb = tf.reshape(emb, (H*W,2))
    if invert_xy:
        emb = emb[...,::-1]
    return emb

class BaseAttentionConvolution(tf.keras.layers.Layer):
    def __init__(self, kernel_size, dilatation, key_dim=None, out_dim=None, out_activation='relu', **kwargs):
        super(BaseAttentionConvolution, self).__init__(**kwargs)

        self.ks = kernel_size
        self.d = dilatation

        kp=(kernel_size-1)//2
        pad = [dilatation*kp,dilatation*kp]
        self.paddings = [[0,0],pad,pad,[0,0]]

        self.shifts = [s*dilatation for s in range(kernel_size)]

        self.key_dim = key_dim
        self.out_dim = out_dim

        if key_dim is not None:
            self.Q_d = tf.keras.layers.Dense(key_dim)
            self.K_d = tf.keras.layers.Dense(key_dim)
        if out_dim is not None:
            self.V_d = tf.keras.layers.Dense(out_dim, activation=out_activation)

    def build(self, input_shape):
        self.H, self.W = input_shape[-3], input_shape[-2]

        padding_mask = tf.pad(tf.zeros((1,self.H, self.W, 1), tf.float32), self.paddings, constant_values=-math.inf)
        self.padding_mask = self.stack_kernel(padding_mask, axis=-1) # [1,H,W,1,ks**2]

    def stack_kernel(self, a, axis=-2):
        return tf.stack([a[:,y:self.H+y,x:self.W+x] for y in self.shifts for x in self.shifts], axis=axis)
    
    def attention_convolution(self, Q, K, V=None, M=None, return_scores=False):
        '''
            Input shapes:
            Q: [B,H,W,D]
            K: [B,H,W,D]
            V: [B,H,W,E]
            M: [B,H,W,ks**2]
        '''
        
        if self.key_dim is not None:
            Q = self.Q_d(Q)
            K = self.K_d(K)
        D = tf.cast(tf.shape(Q)[-1], tf.float32)

        # kernel pixels collection
        K = tf.pad(K, self.paddings)
        K = self.stack_kernel(K, axis=-2) # [B,H,W, ks**2,D]

        # calculate scores
        w = tf.matmul(tf.expand_dims(Q, axis=-2), K, transpose_b=True) + self.padding_mask # [B,H,W,1,ks**2]
        if M is not None:
            w = w + tf.expand_dims(M, axis=-2)
        
        scores = tf.nn.softmax(w/D**0.5)

        if V is not None:
            V = self.stack_kernel(tf.pad(V, self.paddings), axis=-2)
            V = tf.matmul(scores, V)
            V = tf.squeeze(V, axis=-2) # [B,H,W,E]
            if self.out_dim is not None:
                V = self.V_d(V)

        scores = tf.squeeze(scores, axis=-2) # [B,H,W,ks**2]

        if (V is not None) & return_scores:
            return V, scores
        elif V is not None:
            return V
        else:
            return scores


class InitialWeightsAttentionConv(BaseAttentionConvolution):
    def __init__(self, kernel_size, dilatation, key_dim=None, **kwargs):
        super(InitialWeightsAttentionConv, self).__init__(kernel_size=kernel_size,dilatation=dilatation, key_dim=key_dim, **kwargs)

    def call(self, inputs):
        return self.attention_convolution(inputs, inputs, return_scores=True)

    
class MaskedAttentionConv(BaseAttentionConvolution):
    def __init__(self, kernel_size, dilatation, key_dim=None, out_dim=None, out_activation='relu', **kwargs):
        super(MaskedAttentionConv, self).__init__(kernel_size=kernel_size,dilatation=dilatation, key_dim=key_dim, out_dim=out_dim, out_activation=out_activation, **kwargs)

    def call(self, features, mask):
        return self.attention_convolution(Q=features, K=features, V=features, M=mask, return_scores=True)
    
class LearnableKernelMask(tf.keras.layers.Layer):
    def __init__(self, queries_num, kernel_size, **kwargs):
        super(LearnableKernelMask, self).__init__(**kwargs)

        self.Oq = self.add_weight(shape=(1,1,1, queries_num, kernel_size**2), initializer=tf.keras.initializers.GlorotUniform(), trainable=True, name='kernel_mask')

    def call(self, inputs):
        K = tf.expand_dims(inputs, axis=-1)
        score = tf.nn.softmax(tf.matmul(self.Oq, K))
        V = tf.matmul(score, self.Oq, transpose_a=True)
        return inputs+tf.squeeze(V, axis=-2)
    
class ChannelsNorm(tf.keras.layers.Layer):
    def __init__(self, trainable_norm=False, reshape=False, **kwargs):
        super(ChannelsNorm, self).__init__(**kwargs)

        if trainable_norm:
            self.norm = tf.keras.layers.LayerNormalization()

        self.trainable_norm = trainable_norm
        self.reshape = reshape

    def build(self, input_shape):

        if self.reshape:
            H, W, C = input_shape[1], input_shape[2], input_shape[3]
            self.in_reshape = tf.keras.layers.Reshape((H*W,C))
            self.out_reshape = tf.keras.layers.Reshape((H,W,C))

    def call(self, inputs, training=None):
        x = inputs
        if self.reshape:
            x = self.in_reshape(x)

        if self.trainable_norm:
            x = self.norm(x, training=training)
        else:
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            std = tf.math.reduce_std(x, axis=-1, keepdims=True)
            x = (x-mean)/(std+1e-5)

        if self.reshape:
            x = self.out_reshape(x)

        return x
    
class NormalizedEmbeddings(tf.keras.layers.Layer):
    def __init__(self, mid_units, out_units, mid_levels, **kwargs):
        super(NormalizedEmbeddings, self).__init__(**kwargs)
        
        self.in_dense = tf.keras.layers.Dense(mid_units)
        self.mid_denses = tf.keras.Sequential([tf.keras.layers.Dense(mid_units) for _ in range(mid_levels)])
        self.out_dense = tf.keras.layers.Dense(out_units)
        self.norm = ChannelsNorm()

    def call(self, inputs):
        x = self.in_dense(inputs)
        x = self.norm(x)
        x = self.mid_denses(x)
        x = self.out_dense(x)

        return x
    
class MaskedAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, conv_num, kernel_size, dilatation, key_dim, out_dim, activation=None, use_learnable_mask=False, queries_num=16, **kwargs):
        super(MaskedAttentionBlock, self).__init__(**kwargs)

        self.lrn_mask = use_learnable_mask
        if use_learnable_mask:
            self.lrn_mask_layer = LearnableKernelMask(queries_num, kernel_size)

        self.init_mask_layer = InitialWeightsAttentionConv(kernel_size, dilatation, key_dim)
        self.norm = ChannelsNorm()
        if key_dim is not None:
            self.init_dense = tf.keras.layers.Dense(key_dim)
        
        self.convs = [MaskedAttentionConv(kernel_size, dilatation, key_dim, key_dim, out_activation=activation) for _ in range(conv_num)]

    def call(self, inputs):
        X = inputs#self.init_dense(inputs)
        M = self.init_mask_layer(X)
        if self.lrn_mask:
            M = self.lrn_mask_layer(M)
        
        for conv in self.convs:
            x, M = conv(X,M)
            if self.lrn_mask:
                M = self.lrn_mask_layer(M)
            X = self.norm(X+x)

        return X, M
    
class SpatialMemoryLayer(BaseAttentionConvolution):
    def __init__(self, kernel_size, dilatation, key_dim=None, **kwargs):
        super(SpatialMemoryLayer, self).__init__(kernel_size=kernel_size,dilatation=dilatation, key_dim=key_dim, **kwargs)

        kernel_mult = kernel_size//2
        anchors = [[x*kernel_mult,y*kernel_mult] for x in range(3) for y in range(3)]
        masks = tf.stack([[1 if abs(my-y)+abs(mx-x)<=(kernel_mult if ((my!=kernel_mult) | (mx!=kernel_mult)) & ((y!=kernel_mult) | (x!=kernel_mult)) else -1) else 0 for mx in range(kernel_size) for my in range(kernel_size)]
                        for x,y in anchors], axis=0)
        self.masks = tf.cast(masks[tf.newaxis, tf.newaxis, tf.newaxis], tf.float32)
        self.denom = tf.reduce_sum(self.masks, axis=-1)

    def call(self, inputs):
        M = inputs
        Mk = tf.pad(M, self.paddings)
        Mk = self.stack_kernel(Mk, axis=-2)*self.masks #+ self.padding_mask
        Mk = tf.math.divide_no_nan(tf.reduce_sum(Mk, axis=-1),self.denom)

        M = M+M*Mk#tf.nn.softmax(M*Mk*1e1)
        return M
    
class SampleExtractor(tf.keras.layers.Layer):
    def __init__(self, prop_num=16, batch_dims=1, squeeze_input=True, return_idxs=False,**kwargs):
        super(SampleExtractor, self).__init__(**kwargs)

        self.prop_num = prop_num
        self.squeeze_input = squeeze_input
        self.batch_dims = batch_dims
        self.return_idxs = return_idxs

    def build(self, input_shape):

        if self.squeeze_input:
            H, W, C = input_shape[1], input_shape[2], input_shape[3]
            self.in_reshape = tf.keras.layers.Reshape((H*W,C))
            self.mask_reshape = tf.keras.layers.Reshape((H*W,1))
        
    def call(self, inputs, mask=None):

        if self.squeeze_input:
            sources = self.in_reshape(inputs)
            if mask is not None:
                mask = self.mask_reshape(mask)
        else:
            sources = inputs
        sources = tf.expand_dims(sources, axis=-2)

        #initial point
        samples = tf.reduce_mean(sources, axis=-3, keepdims=True) # [B,1,1,D]
        samples_coll = []
        if self.return_idxs:
            samples_idxs = []
        for i in range(self.prop_num):
            dists = tf.reduce_min(tf.reduce_mean(tf.abs(sources-samples), axis=-1), axis=-1)
            if mask is not None:
                dists = dists*mask[...,0]
            idxs = tf.argmax(dists, axis=-1)
            new_sample = tf.expand_dims(tf.gather(sources, idxs, axis=-3, batch_dims=self.batch_dims), axis=-2)
            samples_coll.append(new_sample)
            if self.return_idxs:
                samples_idxs.append(idxs)
            samples = tf.concat(samples_coll, axis=-2)

        samples = tf.squeeze(samples, axis=-3)
        if self.return_idxs:
            samples_idxs = tf.stack(samples_idxs, axis=-1)
            return samples, samples_idxs
        if self.squeeze_input:
            sources = tf.squeeze(sources, axis=-2)
            return samples, sources
        return samples
    
def plot_furthest_points(features, idxs, cols=8, size=3, color='black'):
    pos = LinearPositions(tf.shape(features)[1:-1], flat_output=True, normalize=False, invert_xy=True)

    rows = math.ceil(tf.shape(features)[0]/cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*size, rows*size))
    axs = axs if rows==1 else axs.flat
    for n, ax in enumerate(axs):
        samples_pos = tf.gather(pos, idxs[n], axis=0).numpy()
        ax.imshow(features[n])
        ax.scatter(samples_pos[:,0], samples_pos[:,1], marker='+', color=color)
    
class ExtractPatches(tf.keras.layers.Layer):
    def __init__(self, split_level, **kwargs):
        super(ExtractPatches, self).__init__(**kwargs)

        self.split_level = split_level

    def build(self, input_shape):
        H, W, self.C = input_shape[1], input_shape[2], input_shape[3]
        input_size = max(H,W)

        self.window_size = input_size//self.split_level

    def call(self, inputs):
        P = tf.image.extract_patches(inputs, [1,self.window_size,self.window_size,1], [1,self.window_size,self.window_size,1], [1,1,1,1], 'VALID')
        P = tf.reshape(P, (-1,self.split_level**2,self.window_size**2,self.C))
        return P
    
class ConcatenatePatches(tf.keras.layers.Layer):
    def __init__(self, block_size=2, squeeze=False, **kwargs):
        super(ConcatenatePatches, self).__init__(**kwargs)

        self.block_size = block_size
        self.squeeze = squeeze

    def build(self, input_shape):
        patches, pixels, channels = input_shape[1], input_shape[2], input_shape[3]

        window_dim = int(patches**0.5)
        patch_dim = int(pixels**0.5)

        target_window_dim = window_dim//self.block_size
        target_patch_dim = patch_dim*self.block_size

        self.reconstruct_grids = tf.keras.layers.Reshape((window_dim,window_dim,patch_dim,patch_dim,channels))
        self.permute_grid = tf.keras.layers.Permute([1,3,2,4,5])
        self.concat_windows = tf.keras.layers.Reshape((target_window_dim, target_patch_dim, target_window_dim, target_patch_dim, channels))
        self.unpermute_grid = tf.keras.layers.Permute([1,3,2,4,5])
        self.flatten_grid = tf.keras.layers.Reshape((target_window_dim**2, target_patch_dim**2,channels))
                

    def call(self, inputs):
        x = self.flatten_grid(self.unpermute_grid(self.concat_windows(self.permute_grid(self.reconstruct_grids(inputs)))))
        if self.squeeze:
            return tf.squeeze(x, axis=1)
        return x
    
class SpatialSimilarityFeatures(tf.keras.layers.Layer):
    def __init__(self, 
                 K_neighbors,
                 spatial_features_dim,
                 out_dim,
                 mid_layers,
                 dropout=0.0,
                 mid_activation='relu',
                 scoring_mode='L1',
                 **kwargs
                 ):
        super(SpatialSimilarityFeatures, self).__init__(**kwargs)

        scoring_funcs = {
            'L1': self._l1_score,
            'L2': self._l2_score,
            'Mult': self._mult_score
        }

        self.sf_ffn = FFN(mid_layers, spatial_features_dim*2, spatial_features_dim, dropout, mid_activation)
        self.out_ffn = FFN(mid_layers, out_dim*2, out_dim, dropout, mid_activation)

        self.scoring_func = scoring_funcs[scoring_mode]
        self.K = K_neighbors

    @staticmethod
    def _mult_score(x):
        return tf.matmul(x,x, transpose_b=True)
    
    @staticmethod
    def _l1_score(x):
        return 1-tf.reduce_mean(tf.abs(tf.expand_dims(x, axis=-2)-tf.expand_dims(x,axis=-3)), axis=-1)
    
    @staticmethod
    def _l2_score(x):
        return 1-tf.reduce_mean((tf.expand_dims(x, axis=-2)-tf.expand_dims(x,axis=-3))**2, axis=-1)**0.5
    
    def build(self, input_shape):
        windows, pixels, channels = input_shape[1], input_shape[2], input_shape[3]

        window_dim = int(pixels**0.5)
        pos = LinearPositions((window_dim,window_dim), flat_output=True, normalize=True)
        self.target_pos = pos[tf.newaxis, tf.newaxis, tf.newaxis]
        self.source_pos = pos[tf.newaxis, tf.newaxis,:,tf.newaxis]
        for i, r in enumerate([1,windows,pixels]):
            self.target_pos = tf.repeat(self.target_pos, r, axis=i)
        for i, r in enumerate([1,windows]):
            self.source_pos = tf.repeat(self.source_pos, r, axis=i)

        self.flatten_pos = tf.keras.layers.Reshape((windows,pixels,self.K*2))

    def call(self, inputs, training=None):
        B = tf.shape(inputs)[0]
        w = self.scoring_func(inputs)

        K_scores, K_idxs = tf.math.top_k(w, k=self.K)
        target_pos = tf.repeat(self.target_pos, B, axis=0)
        source_pos = tf.repeat(self.source_pos, B, axis=0)

        K_rel_pos = self.flatten_pos(source_pos-tf.gather(target_pos, K_idxs, axis=-2, batch_dims=3))
        Sf = self.sf_ffn(tf.concat([K_scores, K_rel_pos], axis=-1), training=training)
        
        Sf = self.out_ffn(tf.concat([inputs, Sf], axis=-1), training=training)

        return Sf
    
class PatchSamplesMHA(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads,
                 trainable_norm=True,
                 global_norm=True,
                 **kwargs):
        super(PatchSamplesMHA, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.norm = ChannelsNorm(trainable_norm, reshape=global_norm)

    def build(self, input_shape):
        self.windows, self.pixels, self.channels = input_shape[0][1], input_shape[0][2], input_shape[0][3]
        self.samples_num = input_shape[1][2]

        self.mha = MHA(self.channels, self.channels, self.channels, self.num_heads, False)

    def call(self, inputs, training=None):
        features = tf.reshape(inputs[0], (-1, self.pixels, self.channels))
        samples = tf.reshape(inputs[1], (-1, self.samples_num, self.channels))

        V = self.norm(tf.reshape(features + self.mha(samples, features, samples), (-1,self.windows, self.pixels, self.channels)))

        return V
    

class SamplesMapsExtraction(tf.keras.layers.Layer):
    def __init__(self, samples_num, **kwargs):
        super(SamplesMapsExtraction, self).__init__(**kwargs)

        self.se = SampleExtractor(samples_num, batch_dims=1, squeeze_input=False)

    def call(self, inputs):
        x = tf.squeeze(inputs, 1)
        samples = self.se(x)

        x = tf.matmul(samples, x, transpose_b=True)
        #x = tf.transpose(x, perm=[0,2,1])

        return tf.expand_dims(x, axis=-1)
    

class YUVEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(YUVEncoding, self).__init__(**kwargs)

        self.YUV = tf.constant([[0.299,0.587,0.114],[-0.14713, -0.28886, 0.436],[0.615,-0.51499,-0.10001]])
        self.YUV_B = tf.constant([[[[0.0625, 0.5, 0.5]]]])

    def call(self, inputs):
        return tf.matmul(self.YUV, inputs[...,tf.newaxis])[...,0] + self.YUV_B
    

class MaskedDecoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskedDecoder, self).__init__(**kwargs)

        #self.out_dense = tf.keras.layers.Dense(out_dims)
        self.cross_norm = tf.keras.layers.LayerNormalization()
        self.self_norm = tf.keras.layers.LayerNormalization()
        '''self.conv = tf.keras.Sequential([tf.keras.layers.Reshape((6,32,32)), tf.keras.layers.Permute([2,3,1])] +
                                        [tf.keras.layers.SeparableConv2D(6, 3, padding='same', activation='relu') for i in range(3)]+
                                        [tf.keras.layers.Reshape((6,32**2)), tf.keras.layers.Permute([2,1])])'''

    @staticmethod
    def gen_query_mask(w):
        # w: [B,N,HW]
        w = tf.transpose(w, perm=[0,2,1]) # [B,HW,N]
        mx = tf.reduce_max(w, axis=-1, keepdims=True)+1e-4 # [B,HW,1]
        #w = tf.nn.sigmoid(2*(w/mx-0.5)) # [B,HW,N]: [0,1]
        w = tf.nn.softmax(w/mx)
        w = tf.transpose(w, perm=[0,2,1]) # [B,N,HW]
        w = w-tf.reduce_min(tf.reduce_min(w, axis=-1, keepdims=True), axis=-1, keepdims=True)
        w = w/tf.reduce_max(tf.reduce_max(w, axis=-1, keepdims=1), axis=-1, keepdims=1)
        return w
       
    def call(self, inputs, mask=None):
        Q = inputs[0]
        embs = inputs[1]
        query_mask = inputs[2]
        w = tf.nn.softmax(tf.matmul(Q, embs, transpose_b=True)+query_mask, axis=-1)
        #query_mask = self.conv(query_mask)
        query_mask = self.gen_query_mask(w)
        if mask is not None:
            query_mask = query_mask * mask
        Q = Q + tf.matmul(query_mask, embs)
        Q = self.cross_norm(Q)
        #Q = tf.matmul(query_mask, embs)

        #qw = tf.nn.softmax(tf.matmul(Q, Q, transpose_b=True), axis=-1)
        #Q = Q + tf.matmul(qw, Q)
        #Q = self.self_norm(Q)
        #Q = self.out_dense(Q)
        return Q, query_mask
    
class ZeroQueryMask(tf.keras.layers.Layer):
    def __init__(self, query_num, size, **kwargs):
        super(ZeroQueryMask, self).__init__(**kwargs)

        self.mask = tf.zeros((1,query_num, size))

    def call(self, inputs):
        return self.mask
    

def gen_spatial_embeddings(anchors_num, shape=(32,32)):
    pos = tf.cast(tf.expand_dims(LinearPositions(shape, normalize=True), axis=-2), tf.float32)
    anchors = tf.constant([[x/(anchors_num-1),y/(anchors_num-1)] for x in range(anchors_num) for y in range(anchors_num)], tf.float32)[tf.newaxis, tf.newaxis]

    pos_emb = (2**0.5-tf.reduce_sum((pos-anchors)**2, axis=-1)**0.5)/(2**0.5)
    pos_emb = tf.reshape(pos_emb, (-1,anchors_num**2))
    return pos_emb

class AddPosEmbs(tf.keras.layers.Layer):
    def __init__(self, anchors_num, init_scale=1.0, **kwargs):
        super(AddPosEmbs, self).__init__(**kwargs)

        self.anchors_num = anchors_num
        self.scale = self.add_weight(shape=(), initializer='ones', trainable=True)
        self.init_scale = init_scale

    def build(self, input_shape):
        H,W = input_shape[1], input_shape[2]

        self.pos_emb = gen_spatial_embeddings(self.anchors_num, shape=(H,W))*self.scale*self.init_scale

        self.grid_emb = tf.reshape(self.pos_emb, (1,H,W,self.anchors_num**2))

    def call(self, features, samples=None, idxs=None):
        B = tf.shape(features)[0]
        features_emb = tf.concat([features, tf.repeat(self.grid_emb, B, axis=0)], axis=-1)

        if samples is not None:
            samples_emb = tf.concat([samples,tf.gather(self.pos_emb, idxs, axis=0)], axis=-1)
            return features_emb, samples_emb
        return features_emb
    

class SampleIdxAdjustment(tf.keras.layers.Layer):
    def __init__(self, original_size, strides, **kwargs):
        super(SampleIdxAdjustment, self).__init__(**kwargs)

        self.original_size = original_size
        self.out_size = original_size//strides
        self.strides = strides
    
    def call(self, inputs):
        y = (inputs//self.original_size)
        x = (inputs-y*self.original_size)

        y = y//self.strides
        x = x//self.strides

        return y*self.out_size+x
    
class SqueezeImg(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SqueezeImg, self).__init__(**kwargs)

    def build(self, input_shape):
        preceeding_dims = list(input_shape[1:-3])
        last_dim = [input_shape[-1]]
        squeezed_dim = [input_shape[-3]*input_shape[-2]]

        self.reshape = tf.keras.layers.Reshape(preceeding_dims+squeezed_dim+last_dim)

    def call(self, inputs):
        return self.reshape(inputs)
    
class UnSqueezeImg(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(UnSqueezeImg, self).__init__(**kwargs)

    def build(self, input_shape):
        preceeding_dims = list(input_shape[1:-2])
        last_dim = [input_shape[-1]]
        squeezed_dim = input_shape[-2]
        unsqueezed_dim = [int(squeezed_dim**0.5)]

        self.reshape = tf.keras.layers.Reshape(preceeding_dims+unsqueezed_dim+unsqueezed_dim+last_dim)
        self.calculated_output_shape = preceeding_dims+unsqueezed_dim+unsqueezed_dim+last_dim

    def call(self, inputs):
        return self.reshape(inputs)
    
    def compute_output_shape(self, input_shape):
        return [input_shape[0]] + self.calculated_output_shape
    
class ExtractSampleByIdx(tf.keras.layers.Layer):
    def __init__(self, batch_dims=1, axis=1, **kwargs):
        super(ExtractSampleByIdx, self).__init__(**kwargs)

        self.batch_dims = batch_dims
        self.axis = axis

    def call(self, x, idxs):
        return tf.gather(x, idxs, axis=self.axis, batch_dims=self.batch_dims)
    
class SampleTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, **kwargs):
        super(SampleTransformerEncoder, self).__init__(**kwargs)

        self.extract_sample = ExtractSampleByIdx(batch_dims=1, axis=1)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.pos = SinePositionEncoding(fixed_dims=False)

        self.num_heads = num_heads

    def build(self, input_shape):
        D = input_shape[0][-1]
        self.ffn = FFN(mid_layers=2, mid_units=D*2, output_units=D)

        self.mha = MHA(D, D, D, self.num_heads, softmax_axis=-2)

    def call(self, inputs, training=None):
        x = inputs[0]
        idxs = inputs[1]
        pos_embs = inputs[2]

        V = self.extract_sample(x, idxs)
        Q = x + pos_embs
        K = self.extract_sample(Q, idxs)

        out = self.norm1(self.mha(V,Q,K)+x)
        out = self.norm2(self.ffn(out)+out)

        return out
    
def random_mask_representation(features, labels, num_points=1, single_mask_input=False):
    H,W = features.shape[-3], features.shape[-2]
    if single_mask_input:
        selected_masks = labels
    else:
        selected_masks = tf.transpose(tf.gather(tf.transpose(labels['mask'], [0,3,1,2]), tf.random.categorical(tf.math.log(labels['class']), 1), axis=1, batch_dims=1), [0,2,3,1])
    selected_point = tf.random.categorical(tf.math.log(tf.reshape(tf.nn.avg_pool2d(selected_masks, 2, 1, 'SAME')**2*selected_masks, (len(selected_masks),-1))), num_points)
    selected_point = tf.where(selected_point==H*W, tf.constant(0, dtype=tf.int64), selected_point)
    return (features, selected_point), selected_masks
    
def binarize_masks(features, labels, batch_size=1, size=256, shapes_num=10):
    a = tf.reshape(labels['mask'], (batch_size,size**2,shapes_num))
    global_mask = tf.reduce_max(a, axis=-1, keepdims=True)
    a = tf.where(a==global_mask, 1.0, 0.0)
    global_mask = tf.where(global_mask>0.0, 1.0, 0.0)
    a = tf.reshape(a*global_mask, (batch_size,size,size,shapes_num))
    labels['mask'] = a
    return features, labels


def extract_roi4point(features, selected_point, labels, window_size=32, input_size=256):
    y = selected_point//input_size
    x = selected_point-y*input_size
    # shift point so bbox could fit in image borders
    x_pad = tf.nn.relu(window_size//2-x)
    y_pad = tf.nn.relu(window_size//2-y)
    x = x + x_pad
    y = y + y_pad

    x_pad = tf.nn.relu(x+window_size//2-input_size)
    y_pad = tf.nn.relu(y+window_size//2-input_size)
    x = x - x_pad
    y = y - y_pad

    points_pos = tf.concat([y,x], axis=-1)
    min_pos = points_pos-window_size//2
    max_pos = points_pos+window_size//2
    bboxes = tf.cast(tf.concat([min_pos, max_pos], axis=-1)/input_size, tf.float32)

    box_indices = tf.range(len(features))
    feature_windows = tf.image.crop_and_resize(features, bboxes, box_indices, [window_size,window_size], method='nearest')
    label_windows = tf.image.crop_and_resize(labels, bboxes, box_indices, [window_size,window_size], method='nearest')
    windows_idxs = tf.repeat([[(window_size//2)*(window_size+1)]], len(features), axis=0)

    return (feature_windows, windows_idxs), label_windows