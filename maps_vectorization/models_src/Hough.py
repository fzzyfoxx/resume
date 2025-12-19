import tensorflow as tf
import math
import numpy as np
import cv2 as cv
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
import warnings

import os
import sys
sys.path.append(os.path.abspath("../"))

from models_src.fft_lib import xy_coords, FFT2D, amp_and_phase, top_k2D
from models_src.Attn_variations import SqueezeImg
from models_src.DETR import FFN, MHA

from src.patterns import gen_colors

def random_vector_point(v):
    p0 = v[0]
    p1 = v[1]
    return tf.random.uniform((), 0, 1)*(p1-p0)+p0

def freq_decoder(vec_angle, lengths, size=32):
    XuvP = tf.concat([tf.sin(vec_angle)/lengths*size, tf.cos(vec_angle)/lengths*size], axis=-1)
    XuvP *= tf.sign(XuvP[...,::-1])*tf.sign(XuvP)
    return XuvP

class FreqLinesGenerator():
    def __init__(self, size=32, batch_dims=1):

        self.size = size
        self.M = int(size/2)

        self.xy = xy_coords((size, size))
        for i in range(batch_dims):
            self.xy = self.xy[tf.newaxis]
        self.n = tf.range(1,self.size, dtype=tf.float32)
    
    def random_freqs(self, examples_num, mean=4.0, std=4.0):
        sign = tf.cast(tf.random.uniform((examples_num,2), 0, 2, dtype=tf.int32)*2-1, tf.float32)
        Xuv = tf.clip_by_value(tf.random.normal((examples_num,2), mean, std, dtype=tf.float32), -self.M, self.M)*sign
        lines_num = tf.reduce_sum(tf.abs(Xuv), axis=-1, keepdims=True)
        Xuv = tf.where(lines_num<3, Xuv+tf.sign(Xuv)*2, Xuv)
        phase = tf.random.uniform((examples_num,1), -1.0, 1.0)

        return tf.cast(Xuv, tf.float32)+1e-4, phase
    
    def random_optimized_freqs(self, examples_num, mean=4.0, std=4.0, examples_mult=5):
        diag_mask = xy_coords([examples_num*examples_mult]*2)
        diag_mask = tf.where(diag_mask[...,0]==diag_mask[...,1], 0., 1.)

        sign = tf.cast(tf.random.uniform((examples_num*examples_mult,2), 0, 2, dtype=tf.int32)*2-1, tf.float32)
        Xuv = tf.clip_by_value(tf.random.normal((examples_num*examples_mult,2), mean, std, dtype=tf.float32), -self.M, self.M)*sign
        angles = tf.math.atan2(*tf.split(Xuv, 2, axis=-1))
        diffs = 1-tf.reduce_max((1-tf.abs(tf.sin(angles-tf.transpose(angles, [1,0]))))*diag_mask, axis=-1)
        Xuv = tf.gather(Xuv, tf.math.top_k(diffs, k=examples_num)[1], axis=0)

        lines_num = tf.reduce_sum(tf.abs(Xuv), axis=-1, keepdims=True)
        Xuv = tf.where(lines_num<3, Xuv+tf.sign(Xuv)*2, Xuv)
        phase = tf.random.uniform((examples_num,1), -1.0, 1.0)

        return tf.cast(Xuv, tf.float32)+1e-4, phase
    
    def gen_vecs(self, Xuv, phase, train_output=False):

        angle = self.xy*Xuv[...,tf.newaxis, tf.newaxis,:]/self.size*2*math.pi

        Iuvx = tf.cos(phase[...,tf.newaxis]*math.pi+tf.reduce_sum(angle, axis=-1))

        vecx, vecy = tf.split((32/Xuv), 2, axis=-1)

        vec_angle = tf.math.atan2(vecy,vecx)

        n=tf.range(-self.M,self.M, dtype=tf.float32)

        slope = tf.tan(-vec_angle)#-vecy/vecx
        bias = -slope*vecx*n - vecy*0.5*phase
        y0 = bias
        y0 = tf.clip_by_value(y0, 0, self.size-1)#tf.where(y0<0.0, 0.0, y0)
        x0 = (y0-bias)/slope
        p0 = tf.stack([x0,y0], axis=-1)

        lines_mask = tf.where((x0<0) | (x0>self.size-1), 0.0, 1.0)
        y1 = tf.clip_by_value(slope*(self.size-1)+bias, 0, self.size-1)
        x1 = (y1-bias)/slope

        p1 = tf.stack([x1,y1], axis=-1)
        vecs_col = (tf.stack([p0,p1], axis=-2))*lines_mask[...,tf.newaxis, tf.newaxis] #+shift
        lengths = tf.abs(-slope*vecx)/(slope**2+1)**0.5

        if train_output:
            return vecs_col, lengths, lines_mask, vec_angle
        return vecs_col, lengths, lines_mask, slope, bias, Iuvx, vec_angle
    
class VecDrawer():
    def __init__(self, min_width=0.2, min_num=2, size=32, grayscale=False):

        self.min_width = min_width
        self.min_num = min_num
        self.grayscale = grayscale
        self.size = size

    def cut_vec(self, vec, vec_mask, length, img, color):

        lines_num = tf.cast(tf.reduce_sum(vec_mask), tf.int32)
        filtered_lines_num = tf.random.uniform((), self.min_num, lines_num+1, dtype=tf.int32)

        starting_pos = tf.argmax(vec_mask, axis=-1, output_type=tf.int32, )
        starting_pos = tf.random.uniform((),0, lines_num-filtered_lines_num+1, dtype=tf.int32)+starting_pos
        filtered_vecs = vec[starting_pos:(starting_pos+filtered_lines_num)]

        cutted_vec_mask = tf.pad(tf.ones((len(filtered_vecs),)), [[starting_pos, self.size-starting_pos-filtered_lines_num]])

        vec_slope = (filtered_vecs[:,1,1]-filtered_vecs[:,0,1])/(filtered_vecs[:,1,0]-filtered_vecs[:,0,0])
        vec_bias = filtered_vecs[:,0,1]-vec_slope*filtered_vecs[:,0,0]

        diag_p0 = random_vector_point(filtered_vecs[0])
        diag_pn = random_vector_point(filtered_vecs[-1])

        slope = (diag_pn[1]-diag_p0[1])/(diag_pn[0]-diag_p0[0])
        bias = diag_p0[1]-diag_p0[0]*slope

        diag_x = (bias-vec_bias)/(vec_slope-slope)
        diag_y = vec_slope*diag_x+vec_bias
        vec_diag = tf.stack([diag_x, diag_y], axis=-1)

        vec_side_widths = tf.random.uniform((filtered_lines_num,2,1), self.min_width, 1)
        cutted_vecs = (filtered_vecs-vec_diag[:,tf.newaxis])*vec_side_widths+vec_diag[:,tf.newaxis]

        thickness = np.random.randint(1,max(2,int(length/2//1)))

        img = cv.polylines(img, cutted_vecs.numpy().astype(np.int32), False, color, thickness=thickness)
        mask = cv.polylines(np.zeros((self.size, self.size, 1)), cutted_vecs.numpy().astype(np.int32), False, 1.0, thickness=thickness)

        return img, mask, cutted_vec_mask, tf.pad(cutted_vecs, [[starting_pos, self.size-starting_pos-filtered_lines_num],[0,0],[0,0]]), thickness

    def draw_vecs(self, vecs_col, lines_mask, lengths, colors=None):
        background_color = [clr/255 for clr in gen_colors(grayscale=self.grayscale)]
        img = np.ones((self.size,self.size,3))*np.array([[background_color]], np.float32)
        masks = []
        vec_masks = []
        cutted_vecs_col = []
        thickness_col = []
        if colors is None:
            colors =  [[clr/255 for clr in gen_colors(grayscale=self.grayscale)] for i in range(len(vecs_col))]


        for vec, vec_mask, length, color in zip(vecs_col, lines_mask, lengths, colors):
            img, mask, cutted_vec_mask, cutted_vecs, thickness = self.cut_vec(vec, vec_mask, length, img, color)
            masks.append(tf.constant(mask, tf.float32))
            vec_masks.append(cutted_vec_mask)
            cutted_vecs_col.append(cutted_vecs)
            thickness_col.append(thickness)

        masks = tf.stack(masks+[np.ones((self.size,self.size,1))])
        #masks = tf.concat([1-tf.reduce_max(masks, axis=0, keepdims=True), masks], axis=0)


        return (tf.constant(img, tf.float32), 
                masks, 
                tf.stack(vec_masks+[tf.zeros((self.size,))]), 
                tf.math.floor(tf.stack(cutted_vecs_col+[tf.zeros((self.size,2,2))])), 
                tf.stack(colors+[background_color]),
                tf.constant(thickness_col, tf.float32)
        )


class TopKFreqs(tf.keras.layers.Layer):
    def __init__(self, freq_filter_size=2, pool_size=3, top_k=20, size=32, **kwargs):
        super().__init__(**kwargs)

        self.xy = xy_coords((size, size))
        self.zero_freq_mask = tf.pad(tf.zeros((freq_filter_size,freq_filter_size)), [[0,size-freq_filter_size],[0,size-freq_filter_size]], constant_values=1.)
        self.zero_freq_mask = (self.zero_freq_mask*self.zero_freq_mask[:,::-1])
        self.zero_freq_mask = (self.zero_freq_mask*self.zero_freq_mask[::-1,:])[tf.newaxis]
        self.upper_mask = tf.cast(tf.where(tf.reduce_sum(self.xy, axis=-1)[...,tf.newaxis]>tf.cast(size, tf.float32), 0, 1), tf.float32)[tf.newaxis]

        self.fft = FFT2D()
        self.squeeze = SqueezeImg()

        self.pool_size = pool_size
        self.k = top_k
        self.size = size

    def draw_freqs(self, Xuv, phase):
        angle = self.xy[tf.newaxis,tf.newaxis]*Xuv[:,:,tf.newaxis, tf.newaxis]/self.size*2*math.pi

        Iuvx = tf.cos(phase[:,:,tf.newaxis]*math.pi+tf.reduce_sum(angle, axis=-1))

        return Iuvx

    def call(self, inputs):

        F = FFT2D()(inputs)
        amp, ph = amp_and_phase(F)
        amp = amp/tf.reduce_max(SqueezeImg()(amp), axis=-2, keepdims=True)[:,tf.newaxis]

        amp_arg_max = tf.argmax(amp, axis=-1)[...,tf.newaxis]
        phase_max = tf.gather(ph, amp_arg_max, axis=-1, batch_dims=3)[...,0]

        amp_max = tf.reduce_max(amp*self.upper_mask, axis=-1, keepdims=True)
        amp_pool = tf.nn.max_pool2d(amp_max, self.pool_size, 1, padding='SAME')
        amp_pool = tf.where(amp_max<amp_pool, 0.0, amp_max)

        top_coords, _ = top_k2D(amp_pool[...,0]*self.zero_freq_mask,self.k)

        top_phases = tf.gather_nd(phase_max, top_coords, batch_dims=1)[...,tf.newaxis]

        top_coords = tf.where(top_coords>self.size//2, top_coords-self.size, top_coords)

        Iuvx = self.draw_freqs(tf.cast(top_coords, tf.float32), top_phases)

        return top_phases, top_coords, Iuvx

class FreqShiftOptimization():
    def __init__(self, sample_freq=0.1, sample_range=1., size=32, scoring_func='F1'):

        shifts = tf.range(-sample_range,sample_range+1e-3, sample_freq)
        self.shifts_grid = tf.stack(tf.meshgrid(shifts, shifts), axis=-1)

        self.xy = xy_coords((size, size))
        self.size = size

        self.score_func = self.F1 if scoring_func == 'F1' else self.IoU

        self.props_flg = FreqLinesGenerator(batch_dims=2)

    @staticmethod
    def IoU(a,b):
        return tf.reduce_sum(a*b, axis=-1)/(tf.reduce_sum(a+b, axis=-1)+1e-6)

    @staticmethod
    def F1(y_true,y_pred):
        TP = tf.reduce_sum(y_true*y_pred, axis=-1)
        FP = tf.reduce_sum(y_pred, axis=-1)-TP
        FN = tf.reduce_sum(y_true*(1-y_pred), axis=-1)

        return TP/(TP+(FP+FN)/2)

    def __call__(self, coords, phases, clusters):
        top_Xuv = tf.cast(coords[:,:,tf.newaxis, tf.newaxis], tf.float32)+self.shifts_grid[tf.newaxis, tf.newaxis]+1e-6

        angle = self.xy[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]*top_Xuv[:,:,:,:,tf.newaxis, tf.newaxis]/self.size*2*math.pi
        top_Iuvx = tf.cos(phases[...,tf.newaxis,tf.newaxis,tf.newaxis]*math.pi+tf.reduce_sum(angle, axis=-1))

        a = tf.nn.relu(SqueezeImg()(top_Iuvx[...,tf.newaxis])[...,0])
        b = SqueezeImg()(clusters[:,tf.newaxis, tf.newaxis, tf.newaxis])[...,0]

        scores = self.F1(b,a)

        shift_coords, shift_scores = top_k2D(scores, k=1)

        top_Xuv = tf.gather_nd(top_Xuv, shift_coords[:,:,tf.newaxis,:,::-1], batch_dims=2)[:,:,0,0]
        
        props_vecs_col, props_lengths, props_lines_mask, props_vecs_slope, props_vecs_bias, props_Iuvx, props_vec_angle = self.props_flg.gen_vecs(top_Xuv, phases)

        return props_Iuvx, props_vecs_col, props_lines_mask, top_Xuv
    
def GaussianMixtureComponents(I, max_clusters, max_iter=100):
    X = SqueezeImg()(I[tf.newaxis])[0]
    scores = []
    for i in range(1,max_clusters+1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm = GaussianMixture(n_components=i, max_iter=max_iter).fit(X)
        scores.append(gm.bic(X))

    cl_num = np.argmin(scores)+1
    #print(cl_num)
    gm = GaussianMixture(n_components=cl_num, max_iter=max_iter).fit(X)
    clusters = np.reshape(gm.predict(X), (32,32))

    Cl = tf.stack([tf.where(clusters==i, 1.0, 0.0)[...,tf.newaxis] for i in range(cl_num)])

    return Cl, tf.constant(gm.means_, tf.float32)

def IoU(a,b, axis=None):
    return tf.reduce_sum(a*b, axis=axis)/(tf.reduce_sum(a+b, axis=axis)+1e-6)

def F1(y_true,y_pred, axis=None):
    TP = tf.reduce_sum(y_true*y_pred, axis=axis)
    FP = tf.reduce_sum(y_pred, axis=axis)-TP
    FN = tf.reduce_sum(y_true*(1-y_pred), axis=axis)

    return TP/(TP+(FP+FN)/2)

class VecDataset():
    def __init__(self, flg, batch_size, examples_num, min_width=0.2, min_lines_num=2, parallel_calls=4):

        self.flg = flg
        self.batch_size = batch_size
        self.examples_num = examples_num
        self.vd = VecDrawer(min_width=min_width, min_num=min_lines_num)
        self.parallel_calls = parallel_calls

    def _gen_parameters(self, *args):
        Xuv, phase = self.flg.random_freqs(examples_num=self.examples_num)
        vecs_col, lengths, lines_mask, vecs_slope, vecs_bias, Iuvx, vec_angle = self.flg.gen_vecs(Xuv, phase)

        return vecs_col, lengths, lines_mask, vec_angle, phase
    
    def _map_drawing(self, vecs_col, lengths, lines_mask, vec_angle, phase):
        img = tf.py_function(self.vd.draw_vecs, [vecs_col, lines_mask, lengths], [tf.float32])[0]
        img.set_shape((32,32,3))
        return img, tf.concat([lengths, vec_angle, phase*math.pi], axis=-1)#{'length': lengths, 'angle': vec_angle, 'phase': phase*math.pi}
    
    def new_dataset(self):
        ds = tf.data.Dataset.range(self.batch_size*2)
        ds = ds.map(self._gen_parameters, num_parallel_calls=self.parallel_calls)
        ds = ds.map(self._map_drawing, num_parallel_calls=self.parallel_calls)
        ds = ds.batch(self.batch_size)
        ds = ds.repeat()

        return ds
    

class VecFeaturesLayer(tf.keras.layers.Layer):
    def __init__(self, freq_filter_size=2, pool_size=3, top_k=20, size=32, **kwargs):
        super().__init__(**kwargs)

        xy = xy_coords((size, size))
        self.zero_freq_mask = tf.pad(tf.zeros((freq_filter_size,freq_filter_size)), [[0,size-freq_filter_size],[0,size-freq_filter_size]], constant_values=1.)[tf.newaxis]
        self.upper_mask = tf.cast(tf.where(tf.reduce_sum(xy, axis=-1)[...,tf.newaxis]>tf.cast(size, tf.float32), 0, 1), tf.float32)[tf.newaxis]

        self.fft = FFT2D()
        self.squeeze = SqueezeImg()

        self.pool_size = pool_size
        self.k = top_k
        self.size = size

    def call(self, inputs):

        F = FFT2D()(inputs)
        amp, ph = amp_and_phase(F)
        amp = amp/tf.reduce_max(SqueezeImg()(amp), axis=-2, keepdims=True)[:,tf.newaxis]

        amp_arg_max = tf.argmax(amp, axis=-1)[...,tf.newaxis]
        phase_max = tf.gather(ph, amp_arg_max, axis=-1, batch_dims=3)[...,0]

        amp_max = tf.reduce_max(amp*self.upper_mask, axis=-1, keepdims=True)
        amp_pool = tf.nn.max_pool2d(amp_max, self.pool_size, 1, padding='SAME')
        amp_pool = tf.where(amp_max<amp_pool, 0.0, amp_max)

        top_coords, _ = top_k2D(amp_pool[...,0]*self.zero_freq_mask,self.k)

        top_phases = tf.gather_nd(phase_max, top_coords, batch_dims=1)[...,tf.newaxis]

        top_coords = tf.where(top_coords>self.size//2, top_coords-self.size, top_coords)

        vecx, vecy = tf.split((32/(tf.cast(top_coords, tf.float32)+1e-4)), 2, axis=-1)

        vec_angle = tf.math.atan2(vecy,vecx)

        slope = -vecy/vecx
        lengths = tf.abs(-slope*vecx)/(slope**2+1)**0.5

        OQ_features = tf.concat([lengths, vec_angle, top_phases], axis=-1)

        return OQ_features
    
def gen_shift_vecs(d, vecs_slope):
    shift_vecs_x = (d**2/(vecs_slope**2+1))**0.5
    shift_vecs_y = vecs_slope*shift_vecs_x
    shift_vecs = tf.concat([shift_vecs_x, shift_vecs_y], axis=-1)

    return shift_vecs

def gen_line_keypoints(vecs_slope, vecs_col, lines_mask, d=3, size=32):
    shift_vecs = gen_shift_vecs(d, vecs_slope)

    line_keypoints_num = tf.math.floor((size*2**0.5/d-1)/2)
    keypoints_range = tf.range(-line_keypoints_num, line_keypoints_num+1)

    centroids = tf.reduce_mean(vecs_col, axis=-2, keepdims=True)
    keypoints = (centroids + (shift_vecs[:,tf.newaxis]*keypoints_range[tf.newaxis, :,tf.newaxis])[:,tf.newaxis])*lines_mask[...,tf.newaxis, tf.newaxis]

    return keypoints, shift_vecs


def positional_encoding(length, depth, temperature):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (temperature**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class AddPosEnc(tf.keras.layers.Layer):
    def __init__(self, temperature=10000, **kwargs):
        super().__init__(**kwargs)

        self.temperature = temperature

    def build(self, input_shape):
        self.pos_enc = positional_encoding(input_shape[-2], input_shape[-1], self.temperature)[tf.newaxis]

    def call(self, inputs):
        return inputs+self.pos_enc    

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 attn_dim=512, 
                 key_dim=512, 
                 num_heads=4, 
                 dropout=0.0,
                 FFN_mid_layers=1, 
                 FFN_mid_units=2048,
                 FFN_activation='relu',
                 norm_axis=-2,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.attn_dropout, self.output_dropout = [tf.keras.layers.Dropout(dropout) for _ in range(2)]
        self.attn_addnorm, self.output_addnorm = [tf.keras.Sequential([
            tf.keras.layers.Add(),
            tf.keras.layers.LayerNormalization(axis=norm_axis)])
            for _ in range(2)]

        self.FFN = FFN(FFN_mid_layers, FFN_mid_units, attn_dim, dropout, FFN_activation)

        self.MHA = MHA(attn_dim, attn_dim, key_dim, num_heads)

    def call(self, V, Q=None, K=None, mask=None, training=None):

        if Q is None:
            Q = V

        if K is None:
            K = V

        # Multi-Head-Attention
        V = self.attn_addnorm([V, self.attn_dropout(self.MHA(V, Q, K), training=training)])

        # Feed-Forward-Network
        V = self.output_addnorm([V, self.output_dropout(self.FFN(V), training=training)])

        if mask is not None:
            return V*mask
        return V