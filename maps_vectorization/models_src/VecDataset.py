import os
import numpy as np
import tensorflow as tf

import math
import cv2 as cv

from models_src.fft_lib import xy_coords
from models_src.VecModels import flatten
from src.patterns import gen_colors

from google.cloud import storage
import time

### GENERAL PURPOSE FUNCTIONS ###

def cut_vec(center, angle, size):
    cy, cx = np.split(center, 2, axis=-1)
    slope = np.tan(angle)
    bias = cy-slope*cx
    y0 = bias
    y0 = np.clip(y0, 0, size-1)
    x0 = (y0-bias)/slope
    p0 = np.stack([y0, x0], axis=-1)
    y1 = np.clip(slope*(size-1)+bias, 0, size-1)
    x1 = (y1-bias)/slope

    p1 = np.stack([y1, x1], axis=-1)
    vec = np.stack([p0,p1], axis=-2)

    return vec

def numpy_decode1Dcoords(coords, width, xy=False):
    direction = 1 if xy else -1
    x = coords % width
    y = coords // width
    return np.stack([x,y][::direction], axis=-1)

def numpy_dist_to_line(points, vecs):
    v1, v2 = np.split(vecs, 2, axis=-2)
    y1, x1 = np.split(v1-1e-4, 2, axis=-1)
    y2, x2 = np.split(v2+1e-4, 2, axis=-1)
    y0, x0 = np.split(points, 2, axis=-1)
    
    dist = np.abs((x2-x1)*(y0-y1) - (x0-x1)*(y2-y1))/(((x2-x1)**2 + (y2-y1)**2)**0.5 + 1e-4)
    
    return dist

def gen_point_rot_matrix(rot_angle):
    return np.array([[np.cos(rot_angle), np.sin(rot_angle)],[-np.sin(rot_angle), np.cos(rot_angle)]])

def tan_angle(x):
    return np.arctan(np.tan(x))

def numpy_closest_point_on_line(line_vec, points):

    p1, p2 = np.split(line_vec, 2, axis=-2)
    p0 = points
    vec_angle = np.arctan2(*np.split(p2-p1, 2, axis=-1))
    line_direction = np.concatenate([np.sin(vec_angle), np.cos(vec_angle)], axis=-1)
    vector_to_object = p0-p1
    distance = np.sum(vector_to_object*line_direction, axis=-1, keepdims=True)

    closest_position = p1 + distance*line_direction

    return closest_position



### LINE FILLED ###

class LinesGenerator:
    def __init__(self, size, thickness_range, progressive_thickness, progressive_gamma, min_length, dtype=np.int32, **kwargs):

        self.size = size
        self.thickness_range = thickness_range
        self.progressive_thickness = progressive_thickness
        self.gamma = progressive_gamma
        self.min_length = min_length
        self.dtype = dtype

    def draw_vecs(self, vecs, thickness):
        return np.stack([cv.line(np.zeros((self.size, self.size, 1), self.dtype), *vec, 1, thickness) for vec in vecs.astype(np.int32)], axis=0)
    
    def random_progresive_thickness(self, size=None):
        a = np.arange(*self.thickness_range)
        p = 1/a**self.gamma
        p = p/np.sum(p)
        return np.random.choice(a, size=size, p=p)

    def gen_line_args(self):

        if self.progressive_thickness:
            thickness = self.random_progresive_thickness(size=None)
        else:
            thickness = np.random.randint(*self.thickness_range)

        return {'thickness': thickness}
    
    @staticmethod
    def gen_shift_vec(angle, space, round=True):
        shift_vec =  np.stack([np.sin(angle), np.cos(angle)], axis=-1)*space
        if round:
            shift_vec = np.round(shift_vec)

        return shift_vec
    

class LineFilledGenerator(LinesGenerator):
    def __init__(self, lines_num_range, spacing_range, center_padding, angles_prop_num, **kwargs):
        super().__init__(**kwargs)

        self.lines_num_range = lines_num_range
        self.spacing_range = spacing_range
        self.center_padding = center_padding
        self.angles_prop_num = angles_prop_num

    def gen_line_filled_args(self, max_components, previous_angles=None):

        lines_num = min(np.random.randint(*self.lines_num_range), max_components)
        spacing = np.random.randint(*self.spacing_range)
        center = np.random.randint(self.center_padding, self.size-self.center_padding, 2)

        if previous_angles is not None:
            if len(previous_angles)==0:
                previous_angles = None
        if previous_angles is None:
            angle = np.random.uniform(0., 1.)*math.pi
        else:
            angle = np.random.uniform(0., 1., (self.angles_prop_num,))*math.pi
            angle = angle[np.argmax(np.min(np.abs(np.sin(angle[:,np.newaxis]-previous_angles[np.newaxis])), axis=-1))]

        line_args = self.gen_line_args()

        return {'lines_num': lines_num, 'spacing': spacing, 'center': center, 'angle': angle, **line_args}

    @staticmethod
    def gen_side_vec_centers(shift_vec, vecs_num, center):
        if vecs_num==0:
            return np.transpose(np.array([[],[]], np.float32), axes=[1,0])
        n = np.arange(vecs_num)+1
        return n[:,np.newaxis]*shift_vec[np.newaxis] + center[np.newaxis]

    def gen_vecs_to_draw(self, lines_num, thickness, spacing, angle, center):
        thickness = 1 if thickness==1 else thickness + thickness//2 + 1
        min_length = self.min_length+thickness
        perp_angle = angle-math.pi/2

        perp_vec = cut_vec(center, perp_angle, size=self.size)[0]
        shift_vec = self.gen_shift_vec(perp_angle, thickness+spacing, round=False)
        shift_length = np.sum(shift_vec**2)**0.5
        unit_vec = self.gen_shift_vec(np.arctan(np.tan(angle)), 1.0, round=False)
        unit_vec = np.stack([-unit_vec, unit_vec], axis=0)

        left_side_num, right_side_num = np.sum((center[np.newaxis]-perp_vec)**2, axis=-1)**0.5//shift_length
        left_side_centers = self.gen_side_vec_centers(-shift_vec, left_side_num, center)
        right_side_centers = self.gen_side_vec_centers(shift_vec, right_side_num, center)

        centers = np.concatenate([left_side_centers[::-1], center[np.newaxis], right_side_centers], axis=0)

        vecs = cut_vec(centers, angle, size=self.size)[:,0]
        side_vecs = vecs-centers[:,np.newaxis]
        side_lengths = np.sum(side_vecs**2, axis=-1, keepdims=True)**0.5 - min_length/2
        side_length_filter = np.all(side_lengths[...,0]>0, axis=-1)

        n = len(side_vecs)
        side_draw_lengths = np.random.uniform(0.0, 1., (n,2,1))

        cutted_vecs = (side_draw_lengths*side_lengths+min_length/2)*unit_vec[np.newaxis] + centers[:,np.newaxis]
        cutted_vecs = cutted_vecs[side_length_filter]

        n = len(cutted_vecs)
        lines_to_draw = min(n, lines_num)
        buffer = n-lines_to_draw
        startpoint = np.random.randint(0, buffer+1)
        cutted_vecs = cutted_vecs[startpoint:startpoint+lines_to_draw]

        return np.round(cutted_vecs+0.5,0)
    
    def __call__(self, max_components, previous_angles=None):
        shape_args = self.gen_line_filled_args(max_components=max_components, previous_angles=previous_angles)

        vecs = self.gen_vecs_to_draw(**shape_args)

        shape_masks = self.draw_vecs(vecs[...,::-1], thickness=shape_args['thickness']) if len(vecs)>0 else None
        thickness = (shape_args['thickness']-1)*2+1
        return [(shape_masks, None, vecs, None, tan_angle(np.array([shape_args['angle']])), thickness.astype(np.int32))]
    

class DoubleLineFilledGenerator(LineFilledGenerator):

    @staticmethod
    def shifted_pattern_args(center, angle, spacing, thickness, lines_num, max_components, **kwargs):
        center = center + np.array([np.sin(angle-math.pi/2), np.cos(angle-math.pi/2)])*(spacing+thickness//2)

        lines_num = min(lines_num, max_components)

        return {'lines_num': lines_num, 'spacing': spacing, 'center': center, 'angle': angle, 'thickness': thickness, **kwargs}

    def __call__(self, max_components, previous_angles=None):

        shape_args = self.gen_line_filled_args(max_components=max_components, previous_angles=previous_angles)

        vecs = self.gen_vecs_to_draw(**shape_args)

        shape_masks = self.draw_vecs(vecs[...,::-1], thickness=shape_args['thickness']) if len(vecs)>0 else None
        angle = tan_angle(np.array([shape_args['angle']]))
        thickness = shape_args['thickness']//2*2+1
        outputs = [(shape_masks, None, vecs, None, angle, thickness)]

        max_components -= len(vecs)
        if max_components>0:
            shape_args = self.shifted_pattern_args(**shape_args, max_components=max_components)

            vecs = self.gen_vecs_to_draw(**shape_args)
            shape_masks = self.draw_vecs(vecs[...,::-1], thickness=shape_args['thickness']) if len(vecs)>0 else None

            outputs.append((shape_masks, None, vecs, None, angle, thickness.astype(np.int32)))

        return outputs
    


### POLYLINE ###

class PolylineGenerator(LinesGenerator):
    def __init__(self, vertices_range, min_angle_diff, proposals_num, **kwargs):
        super().__init__(**kwargs)

        self.vertices_range = vertices_range
        self.min_angle_diff = min_angle_diff
        self.proposals_num = proposals_num

    def gen_limited_vec_range(self, point, angle):
        b = np.array([[0,0],[self.size-1, self.size-1]])
        unit_vec = self.gen_shift_vec(angle, 1.0, round=False)

        border_points = np.min(np.max((b-point[...,np.newaxis,:])/unit_vec[...,np.newaxis,:], axis=-2), axis=-1)[...,np.newaxis]*unit_vec + point

        return border_points

    def gen_next_vec(self, startpoint, prev_angle, min_angle_diff, min_length, i, proposals_num):

        if i<=1:
            denom = 1
        else:
            denom = 2
        angle = np.linspace(-math.pi/denom, math.pi*(1-2/proposals_num)/denom, proposals_num)+np.random.uniform(0., math.pi*2/(proposals_num*denom))

        angle += prev_angle
        if i>0:
            prev_angle_filter = np.abs(np.arctan(np.tan(angle-prev_angle)))>min_angle_diff
            angle = angle[prev_angle_filter]

        border_points = self.gen_limited_vec_range(startpoint[np.newaxis], angle)
        prop_vecs_len_filter = np.sum((border_points-startpoint)**2, axis=-1)**0.5>min_length
        border_points = border_points[prop_vecs_len_filter]
        angle = angle[prop_vecs_len_filter]

        if len(border_points)>0:
            point_idx = np.random.randint(0,len(border_points))
            border_point = border_points[point_idx]
            angle = angle[point_idx]
            diff = border_point-startpoint
            vec_length = np.sum(diff**2)**0.5

            normalized_len_buffer = min_length/vec_length
            endpoint = diff * np.random.uniform(normalized_len_buffer, 1.) + startpoint

            return endpoint, angle, np.stack([startpoint, endpoint])
        return None, None, None

    def __call__(self, max_components, **kwargs):

        thickness = self.gen_line_args()['thickness']
        min_length = self.min_length*thickness#+= 0 if thickness==1 else thickness + thickness//2
        vertices_num = min(np.random.randint(*self.vertices_range), max_components)
        startpoint = np.random.randint(0, self.size-1, 2)

        angle = np.random.uniform(-math.pi, math.pi)
        vecs = []
        angles = []
        for i in range(vertices_num):
            startpoint, angle, vec = self.gen_next_vec(startpoint, angle, self.min_angle_diff if i!=1 else self.min_angle_diff * thickness//2+1, min_length,i, self.proposals_num)
            if startpoint is not None:
                vecs.append(vec)
                angles.append(angle)
            else:
                break
        
        vecs = np.round(np.stack(vecs, axis=0)+0.5,0)
        angle = np.stack(angles, axis=0)
        shape_mask = self.draw_vecs(vecs[...,::-1].astype(np.int32), thickness=thickness) if len(vecs)>0 else None
        thickness = (thickness-1)*2+1
        return [(shape_mask, None, vecs, None, angle, thickness.astype(np.int32))]
    

### LINEAR SHAPES ###

class ShapesGenerator:
    def __init__(self, size, size_range, types_probs, filled_prob, samples_per_unit, add_borderline_prob, borderline_thickness_range, min_borderline_thickness_diff, dtype=np.int32, **kwargs):
        self.size = size

        self.size_range = size_range
        self.types_probs = types_probs
        self.filled_prob = filled_prob
        self.samples_per_unit = samples_per_unit
        self.add_borderline_prob = add_borderline_prob
        self.borderline_thickness_range = borderline_thickness_range
        self.min_borderline_thickness_diff = min_borderline_thickness_diff

        self.dtype = dtype

        self.types_probs = np.array(list(types_probs.values()))
        self.types_probs = self.types_probs/np.sum(self.types_probs)
        self.types_names = list(types_probs.keys())

        self.types_func_map = {
            'cross': self.cross_polyline,
            'rectangle': self.rectangle_polyline,
            'triangle': self.triangle_polyline,
            'circle': self.circle_polyline
        }

        self.types_fillability = {
            'cross': False,
            'rectangle': True,
            'triangle': True,
            'circle': True
        }


    @staticmethod
    def cross_polyline(size=1.):
        return np.array([[[1., 0.],[-1., 0.]],[[0.,-1.],[0., 1.]]])*size/2

    @staticmethod
    def rectangle_polyline(size=1.):
        return np.array([[[1., 1.],[-1., 1.]], [[-1., 1.], [-1., -1.]], [[-1., -1.],[1., -1.]], [[1., -1.],[1., 1.]]])*size/2

    @staticmethod
    def triangle_polyline(size=1.):
        return np.array([[[0., -1.], [0., 1.]], [[0., 1.], [1., 0.]], [[1., 0.],[0., -1.]]])*size/2

    def circle_polyline(self, size=1.):
        samples = int(np.mean(size)*self.samples_per_unit)
        angles = np.linspace(-math.pi, math.pi, samples+1)
        points = np.stack([np.sin(angles), np.cos(angles)], axis=-1)*size/2

        return np.stack([points[:-1], points[1:]], axis=-2)
    
    def gen_empty_mask(self):
        return np.zeros((self.size, self.size, 1), dtype=self.dtype)
    
    def draw_filled(self, vecs):
        return cv.fillPoly(self.gen_empty_mask(), pts=vecs[np.newaxis,:,0,::-1], color=1)
    
    def draw_borderline(self, vecs, thickness):
        return cv.polylines(self.gen_empty_mask(), vecs[...,::-1], False, 1, thickness=thickness)
    
    @staticmethod
    def get_shape_bbox(vecs, borderline_buffer, batch_dim=False):
        if batch_dim:
            b = len(vecs)
            vecs = np.reshape(vecs, (b, -1, 2))
        else:
            vecs = np.reshape(vecs, (-1,2))
        x0, x1 = np.min(vecs[...,1], axis=-1)-borderline_buffer, np.max(vecs[...,1], axis=-1)+borderline_buffer
        y0, y1 = np.min(vecs[...,0], axis=-1)-borderline_buffer, np.max(vecs[...,0], axis=-1)+borderline_buffer

        bbox = np.array([[y1, x1],[y0,x1],[y0, x0],[y1, x0]])

        if batch_dim:
            bbox = np.transpose(bbox, axes=[2,0,1])

        return bbox
    
    def gen_shape_args(self):
        shape_type = np.random.choice(self.types_names, p=self.types_probs)

        fillability = self.types_fillability[shape_type]
        filled = np.random.binomial(1, self.filled_prob)*fillability

        shape_size = np.random.randint(*self.size_range)

        borderlined = np.random.binomial(1, self.add_borderline_prob)*filled
        borderline_thickness = min(np.random.randint(*self.borderline_thickness_range), shape_size-self.min_borderline_thickness_diff)
        if borderline_thickness<1:
            if filled:
                borderlined = False
            else:
                borderline_thickness = 1

        vecs = self.types_func_map[shape_type](size=shape_size)

        return {'vecs': vecs, 
                'shape_size': shape_size, 
                'filled': filled, 
                'borderlined': borderlined, 
                'borderline_thickness': borderline_thickness, 
                'shape_type': shape_type
                }
    
    @staticmethod
    def random_rot_matrix():
        angle = np.random.uniform(-math.pi, math.pi)
        return gen_point_rot_matrix(angle), angle
    
    @staticmethod
    def rotate_vecs(vecs, rot_matrix):
        return np.squeeze(np.matmul(rot_matrix[:,np.newaxis, np.newaxis], vecs[...,np.newaxis]), axis=-1)
    
    def shift_and_rotate_vecs(self, vecs, filled, borderlined, borderline_thickness, center, rot_matrix=None, **kwargs):
        if rot_matrix is not None:
            vecs = self.rotate_vecs(vecs, rot_matrix)
        vecs = vecs + center[:,np.newaxis, np.newaxis]
        vecs = np.round(vecs+0.5, 0).astype(np.int64)

        bbox = np.clip(self.get_shape_bbox(vecs, np.any([borderlined, not filled])*(borderline_thickness//2)+0.5, batch_dim=True), 0, self.size-1)

        return vecs, bbox
    
    def gen_shape_mask(self, vecs, filled, borderlined, borderline_thickness, **kwargs):
        
        shape_mask = self.draw_filled(vecs) if filled else self.draw_borderline(vecs, borderline_thickness)

        borderline_mask = (self.draw_borderline(vecs, borderline_thickness) if borderlined else self.gen_empty_mask())#*shape_mask

        return shape_mask, borderline_mask
    
    @staticmethod
    def calc_bordered_shape_size(shape_size, borderlined, borderline_thickness):
        return (shape_size + borderlined*(borderline_thickness//2))*2**0.5
    
    

class LinearShapesGenerator(ShapesGenerator):
    def __init__(self, shapes_num_range, dist_range, angle_proposals_num, **kwargs):
        super().__init__(**kwargs)

        self.shapes_num_range = shapes_num_range
        self.dist_range = dist_range
        self.proposals_num = angle_proposals_num

    def draw_linear_shapes(self, vecs, shape_size, filled, borderlined, borderline_thickness, max_components, **kwargs):

        shapes_num = min(np.random.randint(*self.shapes_num_range), max_components)
        dist = np.random.randint(*self.dist_range)

        rot_matrix, angle = self.random_rot_matrix()
        bordered_shape_size = self.calc_bordered_shape_size(shape_size, np.any([borderlined, not filled]), borderline_thickness)

        dist = max(int(bordered_shape_size/2+1), dist)
        buffer_size = bordered_shape_size + dist
        min_line_length = buffer_size*shapes_num - dist

        center = np.random.randint(0, self.size-1, 2)

        angle = np.linspace(-math.pi/2, math.pi*(1/2-1/self.proposals_num), self.proposals_num)+np.random.uniform(0., math.pi/(2*self.proposals_num))
        border_points = cut_vec(center, angle, self.size)
        border_points_length = np.sum((border_points[:,0]-border_points[:,1])**2, axis=-1)**0.5
       
        prop_vecs_len_filter = border_points_length>min_line_length
        filtered_border_points = border_points[prop_vecs_len_filter]
        filtered_angle = angle[prop_vecs_len_filter]

        if len(filtered_border_points)==0:
            idx = [np.argmax(border_points_length)]
            filtered_border_points = border_points[idx]
            filtered_angle = angle[idx]

        idx = np.random.randint(0, len(filtered_border_points))
        drawing_vec = filtered_border_points[idx]
        angle = filtered_angle[idx]

        drawing_vec_length = np.sum((drawing_vec[1]-drawing_vec[0])**2)**0.5

        shapes_num = int(max(min((drawing_vec_length + dist)//buffer_size, shapes_num), 1))

        buffer_left = drawing_vec_length-shapes_num*buffer_size + dist

        if buffer_left>0:

            shape_buffer = bordered_shape_size/2
            shape_buffer_ratio = shape_buffer/drawing_vec_length
            startpoint_ratio = np.random.uniform(shape_buffer_ratio, buffer_left/drawing_vec_length+shape_buffer_ratio)

            points_ratio = np.linspace(startpoint_ratio, startpoint_ratio+((shapes_num-1)*buffer_size)/drawing_vec_length, shapes_num)
            diff = drawing_vec[1]-drawing_vec[0]
            centers = np.round(points_ratio[:,np.newaxis]*diff[np.newaxis] + drawing_vec[:1] + 0.5, 0)
            
            vecs_col, bboxes = self.shift_and_rotate_vecs(vecs, filled, borderlined, borderline_thickness, centers, rot_matrix[np.newaxis])
            shape_masks, borderline_masks = np.stack([np.stack(self.gen_shape_mask(vecs, filled, borderlined, borderline_thickness), axis=0) for vecs in vecs_col], axis=1)

        else:
            shape_masks, borderline_masks, bboxes = None, None, None

        line_vec = np.stack([centers[0], centers[-1]], axis=0)[np.newaxis]+0.5 if len(centers)>1 else None

        return (shape_masks, borderline_masks, line_vec, bboxes, tan_angle(np.array([angle])), np.int32((shape_size-1)*2))
    
    def __call__(self, max_components, **kwargs):
        return [self.draw_linear_shapes(**self.gen_shape_args(), max_components=max_components)]
    

### SPREADED SHAPES ###

class SpreadedShapesGenerator(ShapesGenerator):
    def __init__(self, shapes_num_range, min_dist, placement_proposals_num, **kwargs):
        super().__init__(**kwargs)

        self.shapes_num_range = shapes_num_range
        self.min_dist = min_dist
        self.placement_proposals_num = placement_proposals_num

    def __call__(self, max_components, **kwargs):

        shapes_num = min(np.random.randint(*self.shapes_num_range), max_components)

        shape_args = self.gen_shape_args()

        vecs, shape_size, filled, borderlined, borderline_thickness = [shape_args[arg] for arg in ['vecs', 'shape_size', 'filled', 'borderlined', 'borderline_thickness']]
        bordered_shape_size = self.calc_bordered_shape_size(shape_size, np.any([borderlined, not filled]), borderline_thickness)

        min_buffer = bordered_shape_size+self.min_dist

        point_range = (int(bordered_shape_size/2)+1, self.size-(int(bordered_shape_size/2)+1))

        centers = np.zeros((0,2))
        
        for i in range(shapes_num):
            if i==0:
                center = np.random.randint(*point_range, 2)
            else:
                proposals = np.random.randint(*point_range, size=(self.placement_proposals_num,2))
                proposals_filter = np.min(np.sum((proposals[:,np.newaxis]-centers[np.newaxis])**2, axis=-1)**0.5, axis=-1)>min_buffer
                proposals = proposals[proposals_filter]

                n = len(proposals)
                if n>0:
                    center = proposals[np.random.randint(0, n)]
                else:
                    break

            centers = np.concatenate([centers, center[np.newaxis]], axis=0)

        shapes_num = len(centers)
        angles = np.random.uniform(-math.pi, math.pi, size=shapes_num)
        rot_matrices = np.transpose(gen_point_rot_matrix(angles), axes=[2,0,1])

        vecs_col, bboxes = self.shift_and_rotate_vecs(vecs, filled, borderlined, borderline_thickness, centers, rot_matrices)
        shape_masks, borderline_masks = np.stack([np.stack(self.gen_shape_mask(vecs, filled, borderlined, borderline_thickness), axis=0) for vecs in vecs_col], axis=1)

        return [(shape_masks, borderline_masks, None, bboxes, None, np.int32((shape_size-1)*2))]
    


### MAP GENERATOR ###
    
class MultishapeMapGenerator:
    def __init__(self, outputs, size, patterns_num_range, max_components_num, color_rand_range, use_previous_color_prob, min_orig_shape_coverage, grayscale, mask_dtype, bbox_vecs,
                 patterns_prob, line_args, shape_args, line_filled_args, polyline_args, linear_shapes_args, spreaded_shapes_args):

        self.size = size

        self.patterns_num_range = patterns_num_range
        self.max_components_num = max_components_num
        self.color_rand_range = color_rand_range
        self.use_previous_color_prob = use_previous_color_prob
        self.min_orig_shape_coverage = min_orig_shape_coverage
        self.grayscale = grayscale
        self.mask_dtype = mask_dtype
        self.max_patterns_num = patterns_num_range[-1]-1

        self.bbox_vecs = bbox_vecs

        self.tf_mask_dtype = tf.as_dtype(mask_dtype)

        self.generator_map = {
            'line_filled': {'generator': LineFilledGenerator(size=size, dtype=mask_dtype, **line_args, **line_filled_args), 'angle_similarity_prevention': True, 'multi_pattern_color_diversity': True, 'single_vec': False, 'type': 'line'},
            'double_line_filled': {'generator': DoubleLineFilledGenerator(size=size, dtype=mask_dtype, **line_args, **line_filled_args), 'angle_similarity_prevention': True, 'multi_pattern_color_diversity': True, 'single_vec': False, 'type': 'line'},
            'polyline': {'generator': PolylineGenerator(size=size, dtype=mask_dtype, **line_args, **polyline_args), 'angle_similarity_prevention': False, 'multi_pattern_color_diversity': False, 'single_vec': False, 'type': 'line'},
            'linear_shapes': {'generator': LinearShapesGenerator(size=size, dtype=mask_dtype, **shape_args, **linear_shapes_args), 'angle_similarity_prevention': False, 'multi_pattern_color_diversity': False, 'single_vec': True, 'type': 'shape'},
            'spreaded_shapes': {'generator': SpreadedShapesGenerator(size=size, dtype=mask_dtype, **shape_args, **spreaded_shapes_args), 'angle_similarity_prevention': False, 'multi_pattern_color_diversity': True, 'single_vec': False, 'type': 'shape'}
        }

        self.patterns_prob = np.array(list(patterns_prob.values()))
        self.patterns_prob = self.patterns_prob/np.sum(self.patterns_prob)
        self.patterns_name = list(patterns_prob.keys())

        self.angle_similarity_prevention_types = ['line_filled', 'double_line_filled']

        self.min_line_length = line_args['min_length']

        self.yx_flat = np.reshape(xy_coords((size, size))[...,::-1].numpy().astype(np.int32), (size**2,2))

        self.all_outputs_info = {
            'img': {'dtype': tf.float32, 'shape': (self.size, self.size, 3), 'name': 'Aimg', 'padded_shape': [self.size, self.size, 3]},
            'angle_label': {'dtype': tf.float32, 'shape': (self.size, self.size, 1), 'name': 'Bangle_label', 'padded_shape': [self.size, self.size, 1]},
            'center_vec_label': {'dtype': tf.float32, 'shape': (self.size, self.size, 2), 'name': 'Ccenter_vec_label', 'padded_shape': [self.size, self.size, 2]},
            'line_label': {'dtype': self.tf_mask_dtype, 'shape': (self.size, self.size, 1), 'name': 'Dline_label', 'padded_shape': [self.size, self.size, 1]},
            'shape_label': {'dtype': self.tf_mask_dtype, 'shape': (self.size, self.size, 1), 'name': 'Eshape_label', 'padded_shape': [self.size, self.size, 1]},
            'thickness_label': {'dtype': tf.int32, 'shape': (self.size, self.size, 1), 'name': 'Fthickness_label', 'padded_shape': [self.size, self.size, 1]},
            'pattern_masks': {'dtype': self.tf_mask_dtype, 'shape': (None, self.size, self.size, 1), 'name': 'Gpattern_masks', 'padded_shape': [self.max_patterns_num, self.size, self.size, 1]},
            'shape_masks': {'dtype': self.tf_mask_dtype, 'shape': (None, self.size, self.size, 1), 'name': 'Hshape_masks', 'padded_shape': [self.max_components_num, self.size, self.size, 1]},
            'vecs_masks': {'dtype': self.tf_mask_dtype, 'shape': (None, self.size, self.size, 1), 'name': 'Ivecs_masks', 'padded_shape': [self.max_components_num, self.size, self.size, 1]},
            'bbox_masks': {'dtype': self.tf_mask_dtype, 'shape': (None, self.size, self.size, 1), 'name': 'Jbbox_masks', 'padded_shape': [self.max_components_num, self.size, self.size, 1]},
            'vecs': {'dtype': tf.float32, 'shape': (None, 2, 2), 'name': 'Kvecs', 'padded_shape': [self.max_components_num, 2,2]},
            'bboxes': {'dtype': tf.float32, 'shape': (None, 2, 2), 'name': 'Lbboxes', 'padded_shape': [self.max_components_num, 4,2]},
            'vecs_mask': {'dtype': self.tf_mask_dtype, 'shape': (None,), 'name': 'Mvecs_mask', 'padded_shape': [self.max_components_num]},
            'bbox_mask': {'dtype': self.tf_mask_dtype, 'shape': (None,), 'name': 'Nbbox_mask', 'padded_shape': [self.max_components_num]},
            'shape_thickness': {'dtype': tf.int32, 'shape': (None,), 'name': 'Oshape_thickness', 'padded_shape': [self.max_components_num]}
        }

        self.outputs = outputs

        self.outputs_info = dict((k,v) for (k,v) in self.all_outputs_info.items() if k in outputs)

        self.output_dtypes, self.output_shapes, self.output_savenames = [self._get_output_info(name) for name in ['dtype', 'shape', 'name']]

        self.output_padded_shapes = dict(((k, v['padded_shape']) for k, v in self.outputs_info.items()))

    def _get_output_info(self, name):
        return [v[name] for (k,v) in self.outputs_info.items()]

    def calc_visible_vec(self, vecs, vec_masks):
        dists_to_line = numpy_dist_to_line(self.yx_flat[np.newaxis], vecs)
        vis_vecs = numpy_decode1Dcoords(np.argmax((tf.reduce_sum((vecs[:,:,tf.newaxis]-self.yx_flat[tf.newaxis, tf.newaxis])**2+1e-4, axis=-1)**0.5 \
                                                    -np.transpose(dists_to_line, axes=[0,2,1])**2)*np.reshape(vec_masks, (-1,1,self.size**2)), axis=-1), width=self.size, xy=False)
        
        return vis_vecs
    
    @staticmethod
    def bbox_from_mask(yx, mask):
        points = yx[mask>0]

        y1, x1 = np.max(points, axis=0)+0.5
        y0, x0 = np.min(points, axis=0)-0.5

        bbox = np.array([[y1, x1],[y0,x1],[y0, x0],[y1, x0]])

        return bbox
    
    def calc_visible_bbox(self, bboxes_masks):
        bboxes_masks = np.reshape(bboxes_masks, (-1, self.size**2))

        bboxes = np.stack([self.bbox_from_mask(self.yx_flat, bbox_mask) for bbox_mask in bboxes_masks], axis=0)

        return bboxes
        

    def calc_visible_part(self, vecs, bboxes, vecs_masks, borderline_masks, mask, angle, single_vec):
        full_vecs_masks = np.max(np.stack([vecs_masks, borderline_masks], axis=0), axis=0) if borderline_masks is not None else vecs_masks
        full_vis_vecs_masks = full_vecs_masks*(1-mask[np.newaxis])
        vis_vecs_masks_filter = np.sum(full_vis_vecs_masks, axis=(1,2,3))>self.min_line_length

        coverage_filter = np.sum(np.reshape(full_vis_vecs_masks, (-1,self.size**2)), axis=-1)/np.sum(np.reshape(full_vecs_masks, (-1,self.size**2)), axis=-1)>self.min_orig_shape_coverage
        vis_vecs_masks_filter *= coverage_filter

        full_vis_vecs_masks = full_vis_vecs_masks[vis_vecs_masks_filter]

        if borderline_masks is not None:
            vis_borderline_masks = borderline_masks[vis_vecs_masks_filter]*full_vis_vecs_masks
            vis_vecs_masks = vecs_masks[vis_vecs_masks_filter]*full_vis_vecs_masks*(1-vis_borderline_masks)
        else:
            vis_borderline_masks = None
            vis_vecs_masks = full_vis_vecs_masks

        vis_vecs = None
        vis_bboxes = None

        if len(full_vis_vecs_masks)>0:
            pattern_mask = np.max(full_vis_vecs_masks, axis=0, keepdims=False)

            if vecs is not None:
                vis_vecs = self.calc_visible_vec(*((vecs[vis_vecs_masks_filter], full_vis_vecs_masks) if not single_vec else (vecs, pattern_mask[np.newaxis])))

            if bboxes is not None:
                vis_bboxes = self.calc_visible_bbox(full_vis_vecs_masks)
                
        else:
            pattern_mask = None

        if angle is not None:
            if len(angle)>1:
                angle = angle[vis_vecs_masks_filter]
        return pattern_mask, full_vis_vecs_masks, vis_vecs_masks, vis_borderline_masks, vis_vecs, vis_bboxes, angle
    
    @staticmethod
    def random_sort(x):
        return x[::(np.random.randint(0,2)*2-1)]
    
    def gen_colors(self, patterns_num):
        colors =  np.reshape(np.array([[clr/255 for clr in gen_colors(grayscale=self.grayscale)] for i in range(patterns_num*2)]), (patterns_num, 2, 3))
        colors = np.concatenate([colors[:1], colors], axis=0)
        previous_color_mask = np.random.binomial(1, self.use_previous_color_prob, (patterns_num,2,1))

        colors = colors[1:]*(1-previous_color_mask) + colors[:-1]*previous_color_mask
        colors = np.array(list(map(self.random_sort, colors)))

        return colors
    
    def gen_single_color_pair(self):
        return np.array([[clr/255 for clr in gen_colors(grayscale=self.grayscale)] for i in range(2)])
    
    @staticmethod
    def get_pattern_drawing(pattern_mask, shape_masks, borderline_masks, shape_color, borderline_color):
        if borderline_masks is None:
            return pattern_mask*shape_color
        
        shape_mask = np.max(shape_masks, axis=0)
        borderline_masks = np.max(borderline_masks, axis=0)

        shape_mask *= (1-borderline_masks)

        return shape_mask*shape_color + borderline_masks*borderline_color
    
    @staticmethod
    def concat_col(col, empty_shape, dtype):
        return np.concatenate(col, axis=0) if len(col)>0 else np.zeros(empty_shape, dtype=dtype)
    
    def get_angle_label(self, shape_masks, angle):
        return np.ma.average(shape_masks*angle[:,np.newaxis, np.newaxis, np.newaxis], weights=shape_masks, axis=0).data
    
    def vec_center_vec(self, vecs, vecs_mask):
        center_vec = numpy_closest_point_on_line(vecs, self.yx_flat[np.newaxis])-self.yx_flat[np.newaxis]
        return np.ma.average(np.reshape(center_vec, (-1,self.size, self.size,2))*vecs_mask, weights=np.repeat(vecs_mask, 2, -1), axis=0).data
    
    def bbox_center_vec(self, bbox, bbox_mask):
        centers = np.mean(bbox, axis=-2, keepdims=True)
        center_vec = centers-self.yx_flat[np.newaxis]
        return np.ma.average(np.reshape(center_vec, (-1,self.size, self.size,2))*bbox_mask, weights=np.repeat(bbox_mask, 2, -1), axis=0).data

    def __call__(self, *args):

        patterns_num = np.random.randint(*self.patterns_num_range)
        #print(patterns_num)
        components_left = self.max_components_num
        previous_angles = np.zeros((0,), dtype=np.float32)

        mask = np.zeros((self.size, self.size, 1), dtype=self.mask_dtype)
        img = np.zeros((self.size, self.size, 3), np.float32)
        angle_label = np.zeros((self.size, self.size, 1), np.float32)
        shape_masks_col = []
        vecs_masks_col = []
        vecs_col = []
        bbox_masks_col = []
        bbox_col = []
        pattern_masks = []
        center_vec_col = []
        thickness_col = []

        pattern_types = np.random.choice(self.patterns_name, size=patterns_num, replace=True, p=self.patterns_prob)
        colors = self.gen_colors(patterns_num)
        #print(colors)

        background_color = np.array([[[clr/255 for clr in gen_colors(grayscale=self.grayscale)]]])

        i=1
        for pattern_type, color in zip(pattern_types, colors):
            #print(pattern_type)
            generator, angle_similarity_prevention, multi_pattern_color_diversity, single_vec, general_type = self.generator_map[pattern_type].values()
            shape_color, borderline_color = color
            j=0

            for shape_masks, borderline_masks, vecs, bboxes, angle, thickness in generator(max_components=components_left, previous_angles=previous_angles):
                if shape_masks is not None:
                    pattern_mask, full_shape_masks, vis_shape_masks, vis_borderline_masks, vis_vecs, vis_bbox, angle = self.calc_visible_part(vecs, bboxes, shape_masks, borderline_masks, mask, angle, single_vec)
                    if multi_pattern_color_diversity & (i>0):
                        shape_color, borderline_color = self.gen_single_color_pair()

                    if pattern_mask is not None:
                        #print(pattern_type)
                        if (general_type=='line'):
                            draw_angle = True
                        elif (general_type=='shape') & (len(vis_bbox)>2):
                            draw_angle = self.bbox_vecs
                        else:
                            draw_angle = False

                        mask += pattern_mask

                        pattern_masks.append(pattern_mask[np.newaxis])
                        shape_masks_col.append(full_shape_masks)

                        if (vis_vecs is not None) & draw_angle:
                            vecs_masks_col.append(full_shape_masks if not single_vec else pattern_mask[np.newaxis])
                            vecs_col.append(vis_vecs.astype(np.float32))

                        if vis_bbox is not None:
                            bbox_masks_col.append(full_shape_masks)
                            bbox_col.append(vis_bbox.astype(np.float32))

                        if general_type=='line':
                            center_vec = self.vec_center_vec(vis_vecs, full_shape_masks)
                            thickness_col.append(np.array([thickness]*len(vis_vecs)))
                        else:
                            center_vec = self.bbox_center_vec(vis_bbox, full_shape_masks)
                            thickness_col.append(np.max(vis_bbox[:, 0]-vis_bbox[:,2], axis=-1))
                        
                        center_vec_col.append(center_vec[np.newaxis])

                        
                        if (angle is not None) & draw_angle:
                            angle_label += self.get_angle_label(full_shape_masks if (not single_vec) | (len(angle)>1) else pattern_mask[np.newaxis], angle)

                        components_left -= len(full_shape_masks)

                        if angle_similarity_prevention:
                            previous_angles = np.concatenate([previous_angles, angle], axis=0)
                        img += self.get_pattern_drawing(pattern_mask, vis_shape_masks, vis_borderline_masks, shape_color, borderline_color)
                    else:
                        #print(pattern_type, 'COVERED')
                        None
                    j+=1
                    i+=1
                if (i>self.max_patterns_num) | (components_left<1):
                    break
            if (i>self.max_patterns_num) | (components_left<1):
                break

        img += (1-mask)*background_color

        pattern_masks, shape_masks, vecs_masks, bbox_masks = [self.concat_col(col, (0,self.size, self.size,1), self.mask_dtype) for col in [pattern_masks, shape_masks_col, vecs_masks_col, bbox_masks_col]]
        vecs, bboxes, center_vec = [self.concat_col(vecs_col, (0,2,2), np.float32), self.concat_col(bbox_col, (0,4,2), np.float32), self.concat_col(center_vec_col, (0,self.size, self.size,2), np.float32)]

        shape_thickness = self.concat_col(thickness_col, (0,), np.int32).astype(np.int32)
        thickness_label = np.max(shape_thickness[:, np.newaxis, np.newaxis, np.newaxis]*shape_masks, axis=0).astype(np.int32) if len(shape_thickness)>0 else np.zeros((self.size, self.size, 1), np.int32)

        line_label = np.max(vecs_masks, axis=0) if len(vecs_masks)>0 else np.zeros((self.size, self.size, 1), dtype=self.mask_dtype)
        shape_label = np.max(bbox_masks, axis=0) if len(bbox_masks)>0 else np.zeros((self.size, self.size, 1), dtype=self.mask_dtype)
        center_vec = np.sum(center_vec, axis=0).astype(np.float32) if len(center_vec)>0 else np.zeros((self.size, self.size, 2), dtype=np.float32)

        vecs_mask = np.ones((len(vecs,)), dtype=self.mask_dtype)
        bbox_mask = np.ones((len(bboxes,)), dtype=self.mask_dtype)
        outputs_mapping =  dict(zip(self.all_outputs_info.keys(), [img, angle_label, center_vec, line_label, shape_label, thickness_label, pattern_masks, shape_masks, vecs_masks, bbox_masks, vecs, bboxes, vecs_mask, bbox_mask, shape_thickness]))

        return [outputs_mapping[k] for k in self.outputs]



### dataset preprocessing funcs ###


@tf.function
def op_line_features(img, line_label, shape_label, angle_label, center_vec_label, thickness_label, **kwargs):
    shape_class = tf.concat([line_label, shape_label], axis=-1)
    all_shapes_mask = tf.reduce_max(shape_class, axis=-1, keepdims=True)
    shape_class = tf.cast(tf.concat([1-all_shapes_mask, shape_class], axis=-1), tf.float32)

    line_label = tf.cast(line_label, tf.float32)
    all_shapes_mask = tf.cast(all_shapes_mask, tf.float32)

    line_label = tf.math.divide_no_nan(line_label, tf.reduce_mean(flatten(line_label), axis=-1)[:, tf.newaxis, tf.newaxis, tf.newaxis])
    thickness_label = tf.cast(thickness_label, tf.float32)
    all_shapes_mask = tf.math.divide_no_nan(all_shapes_mask, tf.reduce_mean(flatten(all_shapes_mask), axis=-1)[:, tf.newaxis, tf.newaxis, tf.newaxis])

    class_mask = tf.ones(tf.shape(shape_class)[:-1], dtype=tf.float32)

    return (img, 
            {'shape_class': shape_class, 'angle': angle_label, 'thickness': thickness_label, 'center_vec': center_vec_label}, 
            {'shape_class': class_mask, 'angle': line_label, 'thickness': all_shapes_mask, 'center_vec': all_shapes_mask}
            )

@tf.function
def blur_img(blur_ratio_range, kernel_size, color_rand_range, img, **kwargs):
    blur_ratio = tf.random.uniform((), *blur_ratio_range)

    img = tf.clip_by_value(img + tf.random.uniform(tf.shape(img), -color_rand_range, color_rand_range), 0., 1.)
    img = tf.nn.avg_pool2d(img, ksize=kernel_size, strides=1, padding='SAME')*blur_ratio + img*(1-blur_ratio)

    return {'img': img, **kwargs}

@tf.function
def random_flip(**kwargs):
    dirs = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)[0]*2-1
    a, b  = dirs[0], dirs[1]
    return dict([(k, v[:,::a, ::b]) for k,v in kwargs.items()])

@tf.function
def op_pixel_similarity(img, pattern_masks, **kwargs):
    background_mask = 1 - tf.reduce_sum(pattern_masks, axis=-4, keepdims=True)
    pattern_masks = tf.concat([background_mask, pattern_masks], axis=-4)
    return (img, tf.cast(pattern_masks, tf.float32))


### DATASET GENERATOR ###


class DatasetGenerator:
    def __init__(self, map_generator, ds_path, fold_size, parallel_calls, padded_batch, output_filter, preprocess_funcs, **kwargs):

        self.fmg = map_generator

        self.ds_path = ds_path
        self.fold_size = fold_size
        self.parallel_calls = parallel_calls
        self.padded_batch = padded_batch

        self.preprocess_funcs = preprocess_funcs

        self.storage_client = storage.Client()

        self.output_filter = output_filter
        self.padded_shapes = dict([(elem, self.fmg.output_padded_shapes[elem]) for elem in output_filter]) if output_filter is not None else self.fmg.output_padded_shapes


    @tf.function
    def _filter_outputs(self, inputs):
        return dict([(elem, inputs[elem]) for elem in self.output_filter])

    @tf.function
    def _gen_images(self, *args):

        inputs = tf.py_function(self.fmg, [], self.fmg.output_padded_shapes)

        return inputs
    
    @tf.function
    def _set_shapes(self, *args):
        # shapes definition
        inputs = args

        for input, input_shape in zip(inputs, self.fmg.output_shapes):
            input.set_shape(input_shape)

        return inputs
    
    def _gen_feature_description(self):
        return {name: tf.io.FixedLenFeature([], tf.string) for name in self.fmg.output_savenames}
    
    @staticmethod
    def create_path_if_needed(path):
        folders = os.path.split(path)
        curr_path = ''
        for folder in folders:
            curr_path = os.path.join(curr_path, folder)
            if not os.path.exists(curr_path):
                os.mkdir(curr_path)

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(self, names, inputs):
        feature = {name: self._bytes_feature(tf.io.serialize_tensor(x)) for name, x in zip(names, inputs)}

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    @staticmethod
    def _parse_function(example_proto, dtypes, feature_description):
        inputs = tf.io.parse_single_example(example_proto, feature_description).values()

        inputs = [tf.io.parse_tensor(x, x_type) for x, x_type in zip(inputs, dtypes)]

        return inputs
    
    def save_tfrec_dataset(self, folds_num=1, starting_num=0):
        self.create_path_if_needed(self.ds_path)
        names = self.fmg.output_savenames
        for fold in range(folds_num):
            ds = self._new_dataset(parallel_calls=1)

            #self.ds = self.ds.map(tf_serialize_example, self.cfg.num_parallel_calls)
            print(f'\n\033[1msaving fold {fold+1}/{folds_num}\033[0m')
            pb = tf.keras.utils.Progbar(self.fold_size)
            with tf.io.TFRecordWriter(f'{self.ds_path}/ds-{fold+starting_num}.tfrec') as writer:
                for inputs in ds:
                    writer.write(self.serialize_example(names,inputs))
                    pb.add(1)

    def upload_dataset_to_storage(self, name):
        bucket_name = self.storage_client.project + name
        ds_files = [(os.path.join(self.ds_path, filename), filename) for filename in os.listdir(self.ds_path)]

        # create if neede and get bucket
        try:
            bucket = self.storage_client.get_bucket(bucket_name)
        except:
            bucket = self.storage_client.create_bucket(self.storage_client.bucket(bucket_name), location='eu')

        pb = tf.keras.utils.Progbar(len(ds_files))
        for filepath, filename in ds_files:
            blob = bucket.blob(filename)
            blob.upload_from_filename(filepath)
            pb.add(1)

    def download_dataset_from_storage(self, name):
        self.create_path_if_needed(self.ds_path)
        bucket_name = self.storage_client.project + name
        bucket = self.storage_client.get_bucket(bucket_name)
        blobs = [blob.name for blob in bucket.list_blobs()]
        ds_files = [(os.path.join(self.ds_path, filename), filename) for filename in blobs]

        pb = tf.keras.utils.Progbar(len(ds_files))
        for filepath, blob_name in ds_files:
            bucket.blob(blob_name).download_to_filename(filepath)
            pb.add(1)

    def delete_bucket(self, name):
        bucket_name = self.storage_client.project + name
        self.storage_client.get_bucket(bucket_name).delete(force=True)

    def dataset_speed_test(self,test_iters, batch_size, ds_iter):
        print('\n\033[1mDataset generator speed test\033[0m')
        start_time = time.time()
        for _ in range(test_iters):
            _ = next(ds_iter)
        proc_time = time.time()-start_time

        print('time per batch: %.3fs | time per example: %.3fs' % (proc_time/test_iters,proc_time/(batch_size*test_iters)))

    @staticmethod
    def delete_dataset(path):
        os.system(f'rm -r {path}')

    def get_ds_sizes(self):
        return ['{}: {:.3f} MB'.format(filename, os.path.getsize(os.path.join(self.ds_path, filename))*1e-6) for filename in os.listdir(self.ds_path)]

    @tf.function
    def _tf_map_drawing(self, *args):
        return tf.numpy_function(self.fmg, [], self.fmg.output_dtypes)

    def _new_dataset(self, parallel_calls):
        ds = tf.data.Dataset.range(self.fold_size)
        ds = ds.map(self._tf_map_drawing, num_parallel_calls=parallel_calls, deterministic=False)

        return ds
    
    def _load_dataset(self, val_idxs, validation):
        feature_description = self._gen_feature_description()
        ds_files = [os.path.join(self.ds_path, filename) for i, filename in enumerate(os.listdir(self.ds_path)) if (i in val_idxs if validation else i not in val_idxs)]
        ds = tf.data.TFRecordDataset(ds_files, num_parallel_reads=self.parallel_calls)

        ds = ds.map(lambda x: self._parse_function(x, self.fmg.output_dtypes, feature_description), num_parallel_calls=self.parallel_calls)

        return ds, len(ds_files)*self.fold_size
    
    def _map_names(self, *args):
        return dict(zip(self.fmg.outputs, args))

    def dataset(self, batch_size=0, repeat=True, from_saved=False, validation=False, val_idxs=[], shuffle_buffer_size=0):
        
        if not from_saved:
            ds = self._new_dataset(self.parallel_calls)
            records = self.fold_size
        else:
            ds, records = self._load_dataset(val_idxs, validation)

        ds = ds.map(self._map_names, num_parallel_calls=self.parallel_calls)

        if self.output_filter is not None:
            ds = ds.map(self._filter_outputs, num_parallel_calls=self.parallel_calls)

        if not validation:
            if shuffle_buffer_size>0:
                ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

        if batch_size>0:
            steps = math.ceil(records/batch_size)
            if not self.padded_batch:
                ds = ds.batch(batch_size)
            else:
                ds = ds.padded_batch(batch_size, padded_shapes=self.padded_shapes)
        else:
            steps = records

        if self.preprocess_funcs is not None:
            for func, func_kwargs, if_validation in self.preprocess_funcs:
                if (not validation) | (validation & if_validation):
                    ds = ds.map(lambda x: func(**x, **func_kwargs), num_parallel_calls=self.parallel_calls)

        if repeat:
            ds = ds.repeat()

        return ds, steps

