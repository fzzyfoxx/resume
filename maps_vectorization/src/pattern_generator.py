import os
import sys
sys.path.append(os.path.abspath("../"))

import numpy as np
from PIL import Image
import json
import tensorflow as tf
import math

from src.patterns import pattern_randomization, drawing_patterns, text_label_randomization, gen_colors
from src.legend import gen_random_legend_properties

def gen_dim_coords(lim, min_space_ratio, extend_ratio=.0):
    space_size = int(lim*min_space_ratio)
    extend_size = int(lim*extend_ratio)
    low_d = -extend_size
    high_d = lim-space_size-1

    d0 = np.clip(np.random.randint(low_d, high_d), 0, None)
    d1 = np.clip(np.random.randint(d0+space_size, lim+extend_size), None, lim)

    return np.array([d0,d1])

def gen_filled_shape_coords(min_width_ratio=0.1, min_height_ratio=0.25, horizontal_pad_range=(0.4, 0.7), vertical_extend_ratio=.5, horizontal_extend_ratio=.5,img_size=(50,100)):
    H, W = img_size
    horizontal_lim = W - int(W*np.random.uniform(*horizontal_pad_range))
    
    xs = np.stack([gen_dim_coords(horizontal_lim, min_width_ratio, extend_ratio=horizontal_extend_ratio) for _ in range(2)], axis=0)
    if np.random.binomial(1, 0.5):
        xs = W - xs

    ys = np.transpose(np.stack([gen_dim_coords(H, min_height_ratio, vertical_extend_ratio) for _ in range(2)], axis=0), axes=[1,0])
    if np.random.binomial(1, 0.5):
        ys = H - ys
    points = np.stack([xs,ys], axis=-1)
    top, bot = points
    points = np.concatenate([top, bot[::-1], top[:1]], axis=0)
    return points[:,np.newaxis]

def gen_line_shape_coords(min_width_ratio=0.25, min_height_ratio=0.25, horizontal_pad_range=(0.4, 0.7), vertical_extend_ratio=.5, horizontal_extend_ratio=.5, img_size=(50,100)):
    H, W = img_size
    if horizontal_pad_range[1]>0:
        horizontal_lim = W - int(W*np.random.uniform(*horizontal_pad_range))
    else:
        horizontal_lim = W
    
    xs = gen_dim_coords(horizontal_lim, min_width_ratio, horizontal_extend_ratio)
    if np.random.binomial(1, 0.5):
        xs = W - xs

    ys = gen_dim_coords(H, min_height_ratio, vertical_extend_ratio)
    if np.random.binomial(1, 0.5):
        ys = H - ys
    points = np.stack([xs,ys], axis=-1)

    return points[:,np.newaxis]


class PatternMatchingGenerator():
    def __init__(self, 
                 output_type,
                 target_y,
                 x_ratio,
                 init_y_range,
                 multiple_dominates_prob,
                 max_dominates_num,
                 map_args_path,
                 shape_args_collection
                 ):
        
        self.target_size = (int(target_y*x_ratio),target_y)
        self.init_y_range = init_y_range
        self.x_ratio = x_ratio
        self.background_type = 'splitted' if output_type in ['pmg1'] else 'plain'
        self.output_type = output_type

        self.multiple_dominates_prob = multiple_dominates_prob
        self.max_dominates_num = max_dominates_num
        self.map_args = self._load_args(map_args_path)

        self.shape_args_matching = {
            'filled': {
                'dominate': shape_args_collection['filled_dominate_shape_args'],
                'noise': shape_args_collection['filled_noise_shape_args']
            },
            'line': {
                'dominate': shape_args_collection['line_dominate_shape_args'],
                'noise': shape_args_collection['line_noise_shape_args']
            }
        }

        self.shape_func_matching = {
            'filled': gen_filled_shape_coords,
            'line': gen_line_shape_coords
        }

    @staticmethod
    def _load_args(path):
        with open(path, "r") as f:
            args = json.loads(f.read())
        return args
    
    def _random_img_size(self):
        y = np.random.randint(*self.init_y_range)
        x = int(self.x_ratio*y)
        return (y,x)
    
    def reload_parcel_inputs(self):
        None

    def _gen_plain_background(self, img_size):
        background_color = np.random.randint(245,256)
        return Image.fromarray(np.ones(list(img_size)[::-1]+[3], dtype=np.uint8)*background_color)
    
    def _gen_splitted_background(self, img_size):
        H, W = img_size
        background = np.concatenate([np.ones((H,W//2,3), np.uint8)*np.array(gen_colors(), np.uint8) for _ in range(2)], axis=1)
        return Image.fromarray(background, mode='RGB')
    
    def _calc_line_angle(self, shape):
        shape=shape[:,0]
        shape = shape[::-1] if shape[0,0]>shape[1,0] else shape
        diffs = (shape[1]-shape[0])
        return np.arctan2(*diffs)/math.pi*180
    
    def _calc_bboxes(self, shapes):
        return np.concatenate([np.concatenate([np.min(shape, axis=0)[:,::-1], np.max(shape, axis=0)[:,::-1]], axis=-1) for 
                                 shape in shapes], axis=0)
    
    def _gen_filtered_masks(self, masks):
        masks = [m.astype(np.float32)/255 for m in masks]
        masks_stack = np.stack(masks, axis=0)
        return np.stack([mask-mask*np.max(masks_stack[i+1:], axis=0) for i, mask in enumerate(masks[:-1])]+[masks[-1]], axis=0)
    
    def gen_full_map(self):
        img_size = self._random_img_size()
        
        #print(img_size)
        legend_properties = gen_random_legend_properties(**self.map_args['random_grid_input'])

        pattern_randomizer = pattern_randomization(legend_properties, **self.map_args['pattern_randomization_args_collection'])
        pattern_randomizer.gen_randomized_pattern_args(legend_properties['patterns_order'][0])
        pattern_styles = pattern_randomizer.pattern_styles
        pattern_styles = [style for style in pattern_styles if style['pattern_style']['pattern_type'] in ['solid', 'striped']] + \
                        [style for style in pattern_styles if style['pattern_style']['pattern_type'] == 'line_filled'] + \
                        [style for style in pattern_styles if style['pattern_style']['pattern_type'] == 'line_border']
        
        solid_patterns = [i for i, style in enumerate(pattern_styles) if style['pattern_style']['pattern_type'] in ['solid', 'striped']]
        transparent_patterns = [i for i, style in enumerate(pattern_styles) if style['pattern_style']['pattern_type'] in ['line_filled', 'line_border']]

        choice_options = solid_patterns[-1:] + transparent_patterns

        verified_max_dominates_num = min(self.max_dominates_num, len(choice_options))
        dominates_num = 1 if not np.random.binomial(1, self.multiple_dominates_prob) else (verified_max_dominates_num if verified_max_dominates_num<=2 else np.random.randint(2, verified_max_dominates_num))

        dominates = np.random.choice(choice_options, dominates_num, False)
        #print(len(dominates), dominates)
        img = self._gen_plain_background(img_size[::-1]) if self.background_type=='plain' else self._gen_splitted_background(img_size[::-1])
        drawer = drawing_patterns(img)
        drawer.set_pattern_size(*img_size[::-1])

        dominate_filled_shape = gen_filled_shape_coords(**self.shape_args_matching['filled']['dominate'], img_size=img_size)
        dominate_line_shape = gen_line_shape_coords(**self.shape_args_matching['line']['dominate'], img_size=img_size)

        shapes = []
        angles = []
        shape_norm = np.array([[img_size[::-1]]])
        for i, info in enumerate(pattern_styles):
            is_filled = 'filled' if info['pattern_style']['pattern_type'] in ['solid', 'striped', 'line_filled'] else 'line'
            is_dominate = 'dominate' if i in dominates else 'noise'

            if is_dominate=='dominate':
                shape = dominate_filled_shape if is_filled == 'filled' else dominate_line_shape
            else:
                func = self.shape_func_matching[is_filled]
                args = self.shape_args_matching[is_filled][is_dominate]
                shape = func(**args, img_size=img_size)
            
            shapes.append(shape/shape_norm)
            drawer.draw_single_pattern(shape, 
                                    info['pattern_style'], 
                                    [0,0], 
                                    info['fill_args'], 
                                    info['line_border_args'],
                                    info['text_label_style'], 
                                    legend_pattern=False, 
                                    shape_only_paste=True, 
                                    transparent_paste= True)
            
            if info['pattern_style']['pattern_type']=='line_border':
                angles.append(self._calc_line_angle(shape))
            elif info['pattern_style']['pattern_type']!='solid':
                angles.append(float(info['fill_args']['angle']))
            else:
                angles.append(0.0)
            
        img = np.array(drawer.img.resize(self.target_size))
        masks = drawer.masks

        if self.output_type=='pmg0':
            #masks = self._gen_filtered_masks(masks)
            masks = np.array([np.array(Image.fromarray(mask.astype(np.uint8), mode='L').resize(self.target_size)).astype(np.float32) for mask in masks])
            bboxes = self._calc_bboxes(shapes)
            return img, masks, angles, shapes, bboxes

        elif self.output_type=='pmg1':
            masks = [m.astype(np.float32)/255 for m in masks]
            masks_stack = np.stack(masks, axis=0)
            filtered_masks = np.stack([mask-mask*np.max(masks_stack[i+1:], axis=0) for i, mask in enumerate(masks[:-1])]+[masks[-1]], axis=0)
            mask = np.max(np.stack(filtered_masks[dominates], axis=0), axis=0)

            mask = np.array(Image.fromarray((np.where(mask>0.1, 1, 0)).astype(np.uint8), mode='L').resize(self.target_size)).astype(np.float32)
            mask = np.where(mask>0.3, 1.0, 0.0).astype(bool)
            return tf.constant(img, tf.float32)/255, tf.constant(mask, tf.bool)
        
        elif self.output_type=='pmg2':
            masks = np.array([np.array(Image.fromarray(mask.astype(np.uint8), mode='L').resize(self.target_size)).astype(np.float32) for mask in masks])
            bboxes = self._calc_bboxes(shapes)

            return tf.constant(masks, tf.float32)/255, tf.constant(angles, tf.float32)[:,tf.newaxis]/180, tf.constant(bboxes, tf.float32)
        
        elif self.output_type=='pmg3':
            masks = np.array([np.array(Image.fromarray(mask.astype(np.uint8), mode='L').resize(self.target_size)).astype(np.float32) for mask in masks])
            perms = np.random.binomial(1, 0.5, len(masks))*2-1
            masks = np.array([np.transpose(mask, axes=[0,1][::perm]) for perm, mask in zip(perms, masks)])
            return tf.where(tf.constant(masks, tf.float32)/255>0.1, 1.0, 0.0)