import os
import sys
sys.path.append(os.path.abspath("../"))

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2 as cv

from src.patterns import gen_colors

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
    def __init__(self, inverse=False, **kwargs):
        super().__init__(**kwargs)
    
        self.func = self.fft if not inverse else self.ifft

    @staticmethod
    def fft(x):
        return tf.signal.fft2d(tf.cast(x, tf.complex64))

    @staticmethod
    def ifft(x):
        return tf.math.real(tf.signal.ifft2d((x)))

    def build(self, input_shape):
        batch_dims_num = len(input_shape)-3
        batch_dims = list(range(batch_dims_num))
        self.in_perm = batch_dims + [d+batch_dims_num for d in [2,0,1]]#[0,3,1,2]#batch_dims + [-1,-3,-2]
        self.out_perm = batch_dims + [d+batch_dims_num for d in [1,2,0]]#[0,2,3,1]#batch_dims + [-2,-1,-3]

    def call(self, inputs):
        x = tf.transpose(inputs, perm=self.in_perm)
        x = self.func(x)
        x = tf.transpose(x, perm=self.out_perm)
    
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        amp, ph = amp_and_phase(inputs)

        return tf.stack([amp,ph], axis=-1)

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