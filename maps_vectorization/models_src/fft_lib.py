import os
import sys
sys.path.append(os.path.abspath("../"))

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2 as cv

from src.patterns import gen_colors
from models_src.DETR import HeadsPermuter, MHA

def amp_and_phase(F):
    Re, Im = tf.math.real(F), tf.math.imag(F)
    amp = (Re**2+Im**2)**0.5
    ph = tf.math.atan2(Im,Re)
    return amp, ph

def xy_coords(shape):
    H, W = shape
    x = tf.repeat(tf.range(0,W)[tf.newaxis], H, axis=0)
    y = tf.repeat(tf.range(0,H)[:,tf.newaxis], W, axis=1)
    xy = tf.cast(tf.stack([x,y], axis=-1), tf.float32)

    return xy

def yx_coords(shape):
    H, W = shape
    x = tf.repeat(tf.range(0,W)[tf.newaxis], H, axis=0)
    y = tf.repeat(tf.range(0,H)[:,tf.newaxis], W, axis=1)
    xy = tf.cast(tf.stack([y,x], axis=-1), tf.float32)

    return xy

def zero_freq_mask(size):
    return tf.pad(tf.constant([[0.]], tf.float32), tf.constant([[0,s-1] for s in size], tf.int32), constant_values=1.)

def channeled_fft(x, inv=False):
    func = tf.signal.fft2d if not inv else tf.signal.ifft2d
    return tf.transpose(func(tf.transpose(x, [0,3,1,2])), [0,2,3,1])

def fft_angles(size):
    yx = yx_coords((size, size))
    yx = tf.where(yx>size/2, yx-size, yx)
    angles = tf.math.atan2(*tf.split(yx, 2, axis=-1))
    return tf.where(angles<0, angles+math.pi, angles)

def argmax_2d(tensor):

  # input format: BxHxWxD
  assert rank(tensor) == 4

  # flatten the Tensor along the height and width axes
  flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))

  # argmax of the flat tensor
  argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

  # convert indexes into 2D coordinates
  argmax_x = argmax // tf.shape(tensor)[2]
  argmax_y = argmax % tf.shape(tensor)[2]

  # stack and return 2D coordinates
  return tf.stack((argmax_x, argmax_y), axis=1)

def rank(tensor):

  # return the rank of a Tensor
  return len(tensor.get_shape())

def decode1Dcoords(coords, width):
   y = coords // width
   x = coords % width

   return tf.stack([x,y], axis=-1)

def decode1Dcoords_yx(coords, width):
   y = coords // width
   x = coords % width

   return tf.stack([y,x], axis=-1)

def encode1Dcoords(coords, width):
    return tf.expand_dims(coords[...,1]*width+coords[...,0], axis=-1)

def top_k2D(x, k=1, channel_dim=False):
    if channel_dim:
      x = tf.reduce_mean(x, axis=-1)
    shape = tf.shape(x)
    x = tf.reshape(x, tf.concat([shape[:-2], [-1]], axis=0))

    values, top_coords = tf.math.top_k(x, k)

    top_coords = decode1Dcoords(top_coords, shape[-1])

    return top_coords, values

def gen_shift_matrix(x,y,shifts_num, shape, xy):
    shift = tf.constant([[[x,y]]], dtype=tf.float32)
    shift_pow = tf.range(shifts_num+1, dtype=tf.float32)[:,tf.newaxis, tf.newaxis, tf.newaxis]

    shift_angle = tf.reduce_sum(shift[tf.newaxis]*shift_pow/np.array([[[shape[::-1]]]])*2*math.pi*xy, axis=-1)
    D = tf.complex(tf.cos(shift_angle), -tf.sin(shift_angle))

    return D

def fft_variables(f):
    F = tf.signal.fft2d(tf.cast(f, tf.complex64))
    Fre, Fim = tf.math.real(F), tf.math.imag(F)
    Amp = tf.math.log((Fre**2+Fim**2)**0.5)
    #Amp /= -tf.reduce_min(Amp)
    Ph = tf.math.atan2(Fim,Fre)
    return F, Fre, Fim, Amp, Ph


def plot_fft(xy, Amp, Ph, color='black', width=0.002, scale=0.75):
    s = len(xy)
    for n in range(s):
        origin = tf.transpose(xy[n], perm=[1,0])
        plt.quiver(*[origin[0],-origin[1]], Amp[n]*tf.math.cos(Ph[n]), Amp[n]*tf.math.sin(Ph[n]), color=color, width=width, scale=scale)
    plt.xlim(-1,s)
    plt.ylim(-s, 1)


def FTcorr2D(a,b, abs_value=True):
    m = a*tf.math.conj(b)
    L = tf.math.real(tf.signal.ifft2d(m/(tf.cast(tf.abs(m), tf.complex64)+1e-12)))

    if abs_value:
       return tf.abs(L)
    return L


def fft_symmetry(Fx, shape=None):
    if shape is None:
        H, W = tf.shape(Fx)[-2:]
    else:
        H, W = shape[0], shape[1]
    xy = xy_coords((H-1,W-1))

    xy = tf.reduce_sum(xy, axis=-1)[tf.newaxis,...,tf.newaxis]
    xy_a = tf.cast(tf.where(xy>tf.cast((H+W)/2-3, tf.float32), 0, 1), tf.float32)
    bot_real = tf.pad((tf.math.real(Fx[:,1:,1:])*xy_a)[:,::-1, ::-1], [[0,0],[1,0],[1,0],[0,0]])
    bot_imag = tf.pad((-tf.math.imag(Fx[:,1:,1:])*xy_a)[:,::-1, ::-1], [[0,0],[1,0],[1,0],[0,0]])
    bot_mask = tf.complex(bot_real, bot_imag)

    top_mask = tf.cast(tf.pad(tf.where(xy>tf.cast((H+W)/2-2, tf.float32), 0.0, 1.0), [[0,0],[1,0],[1,0],[0,0]], constant_values=1.0), tf.complex64)*Fx
    return bot_mask+top_mask


class ProperProposalsGenerator:
    def __init__(self, 
                 proposal_shape, 
                 img_shape, 
                 n_proposals,
                 line_length_range, 
                 shifts_num_range, 
                 shifts_size_range,
                 background_range,
                 thickness_range,
                 proposal_background='zeros',
                 grayscale=False):
        
        self.proposal_shape = proposal_shape
        self.img_shape = img_shape
        self.n_proposals = n_proposals
        self.line_length_range = line_length_range
        self.shifts_num_range = shifts_num_range
        self.shifts_size_range = shifts_size_range
        self.background_range = background_range
        self.thickness_range = thickness_range
        self.proposal_background = proposal_background
        self.grayscale = grayscale


        self.xy = xy_coords(img_shape)

    @staticmethod
    def _calc_center(length, shape):
        return np.array([[np.random.randint(length, s-length) for s in shape[::-1]]])

    def _gen_proposal(self):
        color = [clr/255 for clr in gen_colors(grayscale=self.grayscale)]

        angle = np.random.uniform(0,1)*2*math.pi
        length = np.random.uniform(*self.line_length_range)

        img_center = self._calc_center(length, self.img_shape)

        shifts_num = np.random.randint(*self.shifts_num_range)
        shift = -np.random.randint(*self.shifts_size_range, 2)*np.sign(img_center-np.array(self.img_shape[::-1])/2-0.1)[0]
        #print(shift)
        endpoint = np.array([math.cos(angle), math.sin(angle)])*length
        base_vec = np.round(np.array([-endpoint/2, endpoint/2]), 0).astype(np.int32)
        vec = base_vec+img_center

        proposal_shift = np.round(np.array(self.proposal_shape)//2,0)
        proposal_vec = base_vec+proposal_shift

        thickness = np.random.randint(*self.thickness_range)
        pattern_mask = tf.constant(cv.line(np.zeros(self.img_shape), *vec, color=1, thickness=thickness), dtype=tf.float32)
        if self.proposal_background=='noised':
            background = np.random.uniform(*self.background_range, self.proposal_shape+(3,))
        elif self.proposal_background=='ones':
            background = np.ones(self.proposal_shape+(3,))
        else:
            background = np.zeros(self.proposal_shape+(3,))
        proposal = tf.constant(cv.line(background, *proposal_vec, color=color, thickness=thickness), dtype=tf.float32)

        D = gen_shift_matrix(*shift, shifts_num, self.img_shape, self.xy)

        F1_0 = tf.signal.fft2d(tf.cast(pattern_mask, tf.complex64))
        F1 = tf.reduce_sum(F1_0[tf.newaxis]*D, axis=0)
        img_mask = tf.math.real(tf.signal.ifft2d(F1))[...,tf.newaxis]

        pattern = img_mask*tf.constant([[color]], tf.float32)

        return img_mask, proposal, pattern_mask, pattern
    
    def gen_input(self, output_type='proposals'):
        # proposals: patch with drawed pattern
        # masks: masks of patterns

        img = tf.random.uniform(self.img_shape+(3,), *self.background_range)

        proposals = []

        for i in range(self.n_proposals):
            mask, proposal, _, pattern = self._gen_proposal()
            img = (1-mask)*img + pattern
            if output_type=='proposals':
                proposals.append(proposal)
            elif output_type=='masks':
                proposals.append(mask)

        proposals = tf.stack(proposals, axis=0)

        img = tf.clip_by_value(img, 0.0, 1.0)

        output = [img, proposals]
        if self.grayscale:
            output = [t[...,:1] for t in output]
        return output
    

class Pad2shape2D(tf.keras.layers.Layer):
    def __init__(self, target_shape, pad_value, **kwargs):
        super().__init__(**kwargs)

        self.target_shape = target_shape
        self.pad_value = pad_value

    def build(self, input_shape):

        batch_dims = len(input_shape)-3
        pH, pW = self.target_shape[0]-input_shape[-3], self.target_shape[1]-input_shape[-2]

        self.padding = [[0,0]]*batch_dims + [[0,pH],[0,pW],[0,0]]

    def call(self, inputs):
        return tf.pad(inputs, self.padding, constant_values=self.pad_value)
    

class FFT2D(tf.keras.layers.Layer):
    def __init__(self, inverse=False, return_floats=False, **kwargs):
        super().__init__(**kwargs)
    
        self.func = self.fft if not inverse else self.ifft
        self.return_floats = return_floats if not inverse else False

    @staticmethod
    def fft(x):
        return tf.signal.fft2d(x)

    @staticmethod
    def ifft(x):
        return tf.math.real(tf.signal.ifft2d(x))

    def build(self, input_shape):
        batch_dims_num = len(input_shape)-3
        batch_dims = list(range(batch_dims_num))
        self.in_perm = batch_dims + [d+batch_dims_num for d in [2,0,1]]#[0,3,1,2]#batch_dims + [-1,-3,-2]
        self.out_perm = batch_dims + [d+batch_dims_num for d in [1,2,0]]#[0,2,3,1]#batch_dims + [-2,-1,-3]

    def call(self, Re, Im=None):
        if Im is None:
            x = tf.transpose(Re, perm=self.in_perm)
            x = tf.cast(x, tf.complex64)
        else:
            x = tf.complex(Re, Im)
            x = tf.transpose(x, perm=self.in_perm)
        x = self.func(x)
        x = tf.transpose(x, perm=self.out_perm)

        if self.return_floats:
            return tf.math.real(x), tf.math.imag(x)
        return x
    
class FTcorr2DLayer(tf.keras.layers.Layer):
    def __init__(self, abs_value=True, **kwargs):
        super().__init__(**kwargs)

        self.abs_value = abs_value
        self.ifft = FFT2D(inverse=True)

    def call(self, inputs):
        a, b = inputs[0], inputs[1]

        m = a*tf.math.conj(b)
        L = self.ifft(m/(tf.cast(tf.abs(m), tf.complex64)+1e-12))

        if self.abs_value:
            return tf.abs(L)
        return L


def FTcorr3D(a,b, abs_value=True):
   m = a*tf.math.conj(b)
   L = tf.math.real(tf.signal.ifft3d(m/(tf.cast(tf.abs(m), tf.complex64)+1e-12)))

   if abs_value:
      return tf.abs(L)
   return L


class SqueezeChannels(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.d = tf.concat([[-1],input_shape[1:-2], tf.math.reduce_prod(input_shape[-2:], keepdims=True)], axis=0)

    def call(self, inputs):
        return tf.reshape(inputs, self.d)
    
class AmpPhaseLayer(tf.keras.layers.Layer):
    def __init__(self, stack=False, **kwargs):
        super().__init__(**kwargs)

        self.stack = stack

    def call(self, Re, Im):
        amp = (Re**2+Im**2)**0.5
        ph = tf.math.atan2(Im,Re+1e-6)

        if self.stack:
            return tf.stack([amp,ph], axis=-1)
        return amp, ph

def F_from_freq(Ffreq, axis=-1):
    amp, ph = tf.gather(Ffreq, 0, axis=axis), tf.gather(Ffreq, 1, axis=axis)

    return tf.complex(amp*tf.cos(ph), amp*tf.sin(ph))

class FreqActivation(tf.keras.layers.Layer):
    def __init__(self, complex_axis=-1, mask_input=False, **kwargs):
        super().__init__(**kwargs)

        self.axis = complex_axis
        self.mask_input = mask_input

    def build(self, input_shape):

        self.img_shape = (input_shape[1:3])

    def call(self, inputs):
        amp_raw, ph_raw = tf.gather(inputs, 0, axis=self.axis), tf.gather(inputs, 1, axis=self.axis)

        amp = tf.nn.relu(amp_raw)
        ph = tf.nn.tanh(ph_raw)*math.pi
        Ffreq = tf.stack([amp, ph], axis=self.axis)

        Fraw = F_from_freq(Ffreq, axis=self.axis)

        F = fft_symmetry(Fraw, self.img_shape)

        if self.mask_input:
            mask = tf.nn.sigmoid(tf.gather(inputs, 2, axis=self.axis))
            I = tf.math.real(tf.signal.ifft2d(F))
            F = tf.signal.fft2d(tf.cast(I*mask, tf.complex64))

        return F
    
class FreqSpaceAnglePosEncoding(tf.keras.layers.Layer):
    def __init__(self, embs_dim, size, batch_dims=1, flatten=True,**kwargs):
        super().__init__(**kwargs)

        self.embs_dim = embs_dim
        self.size = size
        self.batch_dims = batch_dims
        self.flatten = flatten

    def build(self, input_shape):

        s = self.embs_dim//2
        ph = math.pi * tf.linspace(1., 3.-2/s, s)[tf.newaxis, tf.newaxis]
        t = tf.range(1, s+1, dtype=tf.float32)[tf.newaxis, tf.newaxis]
        angles_map = fft_angles(self.size)
        
        pos_sin = tf.sin(angles_map*t+ph)
        pos_cos = tf.cos(angles_map*t+ph)
        pos_enc = tf.concat([pos_sin, pos_cos], axis=-1)
        
        if self.flatten:
            pos_enc = tf.reshape(pos_enc, (self.size**2, self.embs_dim))
        
        for _ in range(self.batch_dims):
            pos_enc = pos_enc[tf.newaxis]

        self.pos_enc = pos_enc

    def call(self, inputs=None):
        return self.pos_enc

def complex_conj_matmul(aRe, aIm, bRe, bIm, transpose_a=False, transpose_b=False, real_output=True):
    output = tf.matmul(tf.complex(aRe, aIm), tf.complex(bRe, -bIm), transpose_a=transpose_a, transpose_b=transpose_b)
    if real_output:
        return tf.math.real(output)
    return output

def polar2complex(amp, ph):
    return amp*tf.cos(ph), amp*tf.sin(ph)

class Polar2ComplexLayer(tf.keras.layers.Layer):

    def call(self, amp, ph):
        return polar2complex(amp, ph)

class complexSelfMHA(tf.keras.layers.Layer):
    def __init__(self,
                 emb_dim,
                 num_heads,
                 value_pos_enc=True,
                 single_head_pos_enc=True,
                 return_weights=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.return_weights = return_weights
        self.value_pos_enc = value_pos_enc
        self.single_head_pos_enc = single_head_pos_enc
        self.value_preprocess = True

    def build(self):

        self.Q_d = tf.keras.layers.Dense(self.emb_dim)
        self.K_d = tf.keras.layers.Dense(self.emb_dim)
        if self.value_preprocess:
            self.V_d = tf.keras.layers.Dense(self.emb_dim)
        self.O_d = tf.keras.layers.Dense(self.emb_dim)

        self.denominator = tf.math.sqrt(tf.cast(self.emb_dim, tf.float32))

        self.Q_head_extractior = HeadsPermuter(self.num_heads, reverse=False)
        self.Im_head_extractior = HeadsPermuter(self.num_heads, reverse=False)
        self.K_head_extractior = HeadsPermuter(self.num_heads, reverse=False)
        if self.value_preprocess:
            self.V_head_extractior = HeadsPermuter(self.num_heads, reverse=False)
        self.output_perm = HeadsPermuter(self.num_heads, reverse=True)
        if not self.single_head_pos_enc:
            self.pe_head_extractior = HeadsPermuter(self.num_heads, reverse=False)

    def call(self, Re, Im, pos_enc):
        Q = self.Q_head_extractior(self.Q_d(Re))
        K = self.K_head_extractior(self.K_d(Re))
        Im = self.Im_head_extractior(Im)

        if self.single_head_pos_enc:
            pos_enc = tf.expand_dims(pos_enc, axis=-3)
        else:
            pos_enc = self.pe_head_extractior(pos_enc)

        Q += pos_enc
        K += pos_enc

        scores = complex_conj_matmul(aRe=Q, aIm=Im, bRe=K, bIm=Im, transpose_b=True, real_output=True)/self.denominator
        weights = tf.nn.softmax(scores, axis=-1)

        V = self.V_head_extractior(self.V_d(Re))
        if self.value_pos_enc:
            V += pos_enc
        V = tf.matmul(weights, V)

        V = self.O_d(self.output_perm(V))
        
        if self.return_weights:
            return V, weights
        return V
    

class complexSelfPosEncMHA(complexSelfMHA):

    def build(self):
        self.value_preprocess = False
        self.value_pos_enc = False
        super().build()

    def call(self, Re, Im, pos_enc):
        Q = self.Q_head_extractior(self.Q_d(Re))
        K = self.K_head_extractior(self.K_d(Re))
        Im = self.Im_head_extractior(Im)

        scores = complex_conj_matmul(aRe=Q, aIm=Im, bRe=K, bIm=Im, transpose_b=True, real_output=True)/self.denominator
        weights = tf.nn.softmax(scores, axis=-1)
        
        if self.single_head_pos_enc:
            V = tf.expand_dims(pos_enc, axis=-3)
        else:
            V = self.pe_head_extractior(pos_enc)

        V = tf.matmul(weights, V)

        V = self.O_d(self.output_perm(V))

        if self.return_weights:
            return V, weights
        return V