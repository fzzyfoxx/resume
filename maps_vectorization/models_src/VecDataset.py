"""VecDataset module: utilities for synthetic vectorized map generation, labeling, and TensorFlow dataset pipelines.

This module provides geometry helpers, procedural generators for lines and shapes,
label construction routines for training, and a DatasetGenerator to produce
TFRecord-backed datasets.
"""
import os
import numpy as np
import tensorflow as tf

import math
import cv2 as cv

from models_src.fft_lib import xy_coords, decode1Dcoords, fft_angles
from models_src.VecModels import flatten, calc_2x2_vec_angle, two_side_angle_diff, gen_rot_matrix_yx
from src.patterns import gen_colors

from google.cloud import storage
import time

### GENERAL PURPOSE FUNCTIONS ###

def cut_vec(center, angle, size):
    """Clip the infinite line passing through a center at angle to image bounds.

    Args:
        center: [N, 2] or [2] array-like in (y, x) order.
        angle: scalar or [N] angle in radians (0 along +x with tan convention).
        size: int image size (square image with coordinates [0, size-1]).

    Returns:
        vec: [N, 2, 2] endpoints (y, x) of the clipped segment across the image.
    """
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
    """Decode flat indices to 2D coordinates.

    Args:
        coords: np.ndarray of flat indices.
        width: image width.
        xy: if True return (x, y); otherwise (y, x).

    Returns:
        np.ndarray of shape (..., 2) with decoded coordinates.
    """
    direction = 1 if xy else -1
    x = coords % width
    y = coords // width
    return np.stack([x,y][::direction], axis=-1)

def numpy_dist_to_line(points, vecs):
    """Compute perpendicular distance from points to line segments.

    Args:
        points: [..., 2] (y, x) points.
        vecs: [..., 2, 2] (y, x) endpoints of segments.

    Returns:
        Distance array broadcast over inputs with same batch shape as points vs vecs.
    """
    v1, v2 = np.split(vecs, 2, axis=-2)
    y1, x1 = np.split(v1-1e-4, 2, axis=-1)
    y2, x2 = np.split(v2+1e-4, 2, axis=-1)
    y0, x0 = np.split(points, 2, axis=-1)
    
    dist = np.abs((x2-x1)*(y0-y1) - (x0-x1)*(y2-y1))/(((x2-x1)**2 + (y2-y1)**2)**0.5 + 1e-4)
    
    return dist

def gen_point_rot_matrix(rot_angle):
    """Generate 2x2 rotation matrix for rotating (y, x) points by rot_angle.

    Note: The matrix is constructed for (y, x) ordering consistent with module.

    Args:
        rot_angle: scalar angle in radians.

    Returns:
        np.ndarray 2x2 rotation matrix.
    """
    return np.array([[np.cos(rot_angle), np.sin(rot_angle)],[-np.sin(rot_angle), np.cos(rot_angle)]])

def tan_angle(x):
    """Map angle to principal value using tan/atan to keep orientation equivalence.

    Args:
        x: angle in radians.

    Returns:
        Angle wrapped to (-pi/2, pi/2).
    """
    return np.arctan(np.tan(x))

def numpy_closest_point_on_line(line_vec, points):
    """Project points orthogonally onto a line segment.

    Args:
        line_vec: [..., 2, 2] line endpoints (y, x).
        points: [..., 2] query points (y, x).

    Returns:
        [..., 2] closest positions on the infinite line through the segment.
    """

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
    """Base class for generating rasterized line-like patterns.

    Handles size, thickness sampling, and drawing utilities used by subclasses.
    """
    def __init__(self, size, thickness_range, progressive_thickness, progressive_gamma, min_length, dtype=np.int32, **kwargs):
        """Initialize generator configuration.

        Args:
            size: output square size.
            thickness_range: (low, high) integer pixel thickness range.
            progressive_thickness: if True, sample thickness with power-law.
            progressive_gamma: gamma used for progressive thickness distribution.
            min_length: minimal drawable length in pixels.
            dtype: mask dtype used by OpenCV drawing ops.
        """

        self.size = size
        self.thickness_range = thickness_range
        self.progressive_thickness = progressive_thickness
        self.gamma = progressive_gamma
        self.min_length = min_length
        self.dtype = dtype

    def draw_vecs(self, vecs, thickness):
        """Rasterize multiple segments to masks using OpenCV.

        Args:
            vecs: [N, 2, 2] (y, x) endpoints.
            thickness: integer line thickness.
        Returns:
            [N, H, W, 1] mask array with lines drawn.
        """
        return np.stack([cv.line(np.zeros((self.size, self.size, 1), self.dtype), *vec, 1, thickness) for vec in vecs.astype(np.int32)], axis=0)
    
    def random_progresive_thickness(self, size=None):
        """Sample thicknesses with power-law distribution controlled by gamma.
        
        Args:
            size: number of samples to draw.
        Returns:
            Sampled thicknesses as int or np.ndarray.
        """
        a = np.arange(*self.thickness_range)
        p = 1/a**self.gamma
        p = p/np.sum(p)
        return np.random.choice(a, size=size, p=p)

    def gen_line_args(self):
        """Sample line drawing arguments (currently thickness only)."""

        if self.progressive_thickness:
            thickness = self.random_progresive_thickness(size=None)
        else:
            thickness = np.random.randint(*self.thickness_range)

        return {'thickness': thickness}
    
    @staticmethod
    def gen_shift_vec(angle, space, round=True):
        """Generate a perpendicular shift vector of given spacing.

        Args:
            angle: base angle of primary line.
            space: distance to shift.
            round: if True, round components for pixel shifts.
        Returns:
            [2] shift vector in (y, x).
        """
        shift_vec =  np.stack([np.sin(angle), np.cos(angle)], axis=-1)*space
        if round:
            shift_vec = np.round(shift_vec)

        return shift_vec
    

class LineFilledGenerator(LinesGenerator):
    """Generate a bundle of parallel clipped line segments filling a corridor."""
    def __init__(self, lines_num_range, spacing_range, center_padding, angles_prop_num, **kwargs):
        """Configure multi-line generator.

        Args:
            lines_num_range: (low, high) number of lines to draw.
            spacing_range: (low, high) spacing in pixels between lines.
            center_padding: min distance of center from borders.
            angles_prop_num: proposals when avoiding previous angles.
        """
        super().__init__(**kwargs)

        self.lines_num_range = lines_num_range
        self.spacing_range = spacing_range
        self.center_padding = center_padding
        self.angles_prop_num = angles_prop_num

    def gen_line_filled_args(self, max_components, previous_angles=None):
        """Sample parameters for a line-filled pattern, respecting constraints.

        Args:
            max_components: limit on number of drawable components.
            previous_angles: optional angles to avoid for diversity.
        Returns:
            Dict of generation args including lines_num, spacing, center, angle, thickness.
        """

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
        """Generate centers for lines offset from the main center.

        Args:
            shift_vec: [2] shift per line.
            vecs_num: number of centers to generate.
            center: [2] base center.
        Returns:
            [vecs_num, 2] array of centers.
        """
        if vecs_num==0:
            return np.transpose(np.array([[],[]], np.float32), axes=[1,0])
        n = np.arange(vecs_num)+1
        return n[:,np.newaxis]*shift_vec[np.newaxis] + center[np.newaxis]

    def gen_vecs_to_draw(self, lines_num, thickness, spacing, angle, center):
        """Construct and clip the actual line segments to draw.

        Returns:
            [M, 2, 2] array of (y, x) endpoints, possibly fewer than lines_num.
        """
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
        """Generate a single line-filled pattern instance.

        Returns:
            List with a tuple: (shape_masks, borderline_masks, vecs, bboxes, angle, thickness).
        """
        shape_args = self.gen_line_filled_args(max_components=max_components, previous_angles=previous_angles)

        vecs = self.gen_vecs_to_draw(**shape_args)

        shape_masks = self.draw_vecs(vecs[...,::-1], thickness=shape_args['thickness']) if len(vecs)>0 else None
        thickness = np.array((shape_args['thickness']-1)*2+1)
        return [(shape_masks, None, vecs, None, tan_angle(np.array([shape_args['angle']])), thickness.astype(np.int32))]
    

class DoubleLineFilledGenerator(LineFilledGenerator):
    """Draw two parallel line-filled bands separated by spacing."""

    @staticmethod
    def shifted_pattern_args(center, angle, spacing, thickness, lines_num, max_components, **kwargs):
        """Shift the center perpendicular to create a second parallel band."""
        center = center + np.array([np.sin(angle-math.pi/2), np.cos(angle-math.pi/2)])*(spacing+thickness//2)

        lines_num = min(lines_num, max_components)

        return {'lines_num': lines_num, 'spacing': spacing, 'center': center, 'angle': angle, 'thickness': thickness, **kwargs}

    def __call__(self, max_components, previous_angles=None):
        """Generate up to two adjacent line-filled patterns."""

        shape_args = self.gen_line_filled_args(max_components=max_components, previous_angles=previous_angles)

        vecs = self.gen_vecs_to_draw(**shape_args)

        shape_masks = self.draw_vecs(vecs[...,::-1], thickness=shape_args['thickness']) if len(vecs)>0 else None
        angle = tan_angle(np.array([shape_args['angle']]))
        thickness = np.array(shape_args['thickness']//2*2+1)
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
    """Generate random polylines with angle and length constraints."""
    def __init__(self, vertices_range, min_angle_diff, proposals_num, **kwargs):
        """Configure polyline generation.

        Args:
            vertices_range: (low, high) number of segments.
            min_angle_diff: minimal angle difference between consecutive segments.
            proposals_num: candidate angles per step.
        """
        super().__init__(**kwargs)

        self.vertices_range = vertices_range
        self.min_angle_diff = min_angle_diff
        self.proposals_num = proposals_num

    def gen_limited_vec_range(self, point, angle):
        """Compute border-intersection points for a ray from point at angle.

        Returns:
            [2] endpoints clipped to image bounds.
        """
        b = np.array([[0,0],[self.size-1, self.size-1]])
        unit_vec = self.gen_shift_vec(angle, 1.0, round=False)

        border_points = np.min(np.max((b-point[...,np.newaxis,:])/unit_vec[...,np.newaxis,:], axis=-2), axis=-1)[...,np.newaxis]*unit_vec + point

        return border_points

    def gen_next_vec(self, startpoint, prev_angle, min_angle_diff, min_length, i, proposals_num):
        """Propose and pick the next segment endpoint satisfying constraints."""

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
        """Generate a random polyline and its rasterized mask.

        Returns:
            List with tuple (mask, None, vecs, None, angles, thickness).
        """

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
        thickness = np.array((thickness-1)*2+1)
        return [(shape_mask, None, vecs, None, angle, thickness.astype(np.int32))]
    

### LINEAR SHAPES ###

class ShapesGenerator:
    """Abstract shape utilities for filled/borderline shapes and transforms."""
    def __init__(self, size, size_range, types_probs, filled_prob, samples_per_unit, add_borderline_prob, borderline_thickness_range, min_borderline_thickness_diff, dtype=np.int32, **kwargs):
        """Configure generic shape drawing and sampling parameters."""
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
        """Return a cross polyline of given normalized size centered at origin."""
        return np.array([[[1., 0.],[-1., 0.]],[[0.,-1.],[0., 1.]]])*size/2

    @staticmethod
    def rectangle_polyline(size=1.):
        """Return rectangle polyline (axis-aligned) centered at origin."""
        return np.array([[[1., 1.],[-1., 1.]], [[-1., 1.], [-1., -1.]], [[-1., -1.],[1., -1.]], [[1., -1.],[1., 1.]]])*size/2

    @staticmethod
    def triangle_polyline(size=1.):
        """Return an isosceles right triangle polyline centered at origin."""
        return np.array([[[0., -1.], [0., 1.]], [[0., 1.], [1., 0.]], [[1., 0.],[0., -1.]]])*size/2

    def circle_polyline(self, size=1.):
        """Approximate a circle with piecewise linear segments.

        Args:
            size: diameter in normalized units.
        Returns:
            [M, 2, 2] segments approximating the circle.
        """
        samples = int(np.mean(size)*self.samples_per_unit)
        angles = np.linspace(-math.pi, math.pi, samples+1)
        points = np.stack([np.sin(angles), np.cos(angles)], axis=-1)*size/2

        return np.stack([points[:-1], points[1:]], axis=-2)
    
    def gen_empty_mask(self):
        """Create an empty mask array of current size and dtype."""
        return np.zeros((self.size, self.size, 1), dtype=self.dtype)
    
    def draw_filled(self, vecs):
        """Fill a polygon defined by ordered vertices of vecs (y, x)."""
        return cv.fillPoly(self.gen_empty_mask(), pts=vecs[np.newaxis,:,0,::-1], color=1)
    
    def draw_borderline(self, vecs, thickness):
        """Draw polygon outline with specified thickness."""
        return cv.polylines(self.gen_empty_mask(), vecs[...,::-1], False, 1, thickness=thickness)
    
    @staticmethod
    def get_shape_bbox(vecs, borderline_buffer, batch_dim=False):
        """Compute axis-aligned bounding boxes around shape vertices.

        Args:
            vecs: [..., 2, 2] vertices (y, x).
            borderline_buffer: extra pixels to include around bbox.
            batch_dim: if True, preserve batch leading dimension.
        Returns:
            BBoxes formatted as [[y1,x1],[y0,x1],[y0,x0],[y1,x0]].
        """
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
        """Sample a random shape type, size, and its drawing flags.

        Returns:
            Dict containing vecs, shape_size, filled, borderlined, thickness, shape_type.
        """
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
        """Sample a random rotation matrix and return it with the angle."""
        angle = np.random.uniform(-math.pi, math.pi)
        return gen_point_rot_matrix(angle), angle
    
    @staticmethod
    def rotate_vecs(vecs, rot_matrix):
        """Apply rotation matrix to shape vecs (supports batch of matrices)."""
        return np.squeeze(np.matmul(rot_matrix[:,np.newaxis, np.newaxis], vecs[...,np.newaxis]), axis=-1)
    
    def shift_and_rotate_vecs(self, vecs, filled, borderlined, borderline_thickness, center, rot_matrix=None, **kwargs):
        """Rotate and shift shape vecs to centers; also compute clipped bboxes.

        Args:
            vecs: base shape segments around origin.
            center: [N,2] centers to shift to.
            rot_matrix: [N,2,2] or [1,2,2] rotation matrices or None.
        Returns:
            (vecs, bboxes) with integer-rounded coordinates and clipped bboxes.
        """
        if rot_matrix is not None:
            vecs = self.rotate_vecs(vecs, rot_matrix)
        vecs = vecs + center[:,np.newaxis, np.newaxis]
        vecs = np.round(vecs+0.5, 0).astype(np.int64)

        bbox = np.clip(self.get_shape_bbox(vecs, np.any([borderlined, not filled])*(borderline_thickness//2)+0.5, batch_dim=True), 0, self.size-1)

        return vecs, bbox
    
    def gen_shape_mask(self, vecs, filled, borderlined, borderline_thickness, **kwargs):
        """Rasterize filled shape and optional borderline mask."""
        
        shape_mask = self.draw_filled(vecs) if filled else self.draw_borderline(vecs, borderline_thickness)

        borderline_mask = (self.draw_borderline(vecs, borderline_thickness) if borderlined else self.gen_empty_mask())#*shape_mask

        return shape_mask, borderline_mask
    
    @staticmethod
    def calc_bordered_shape_size(shape_size, borderlined, borderline_thickness):
        """Approximate diagonal size for spacing when borderline is present."""
        return (shape_size + borderlined*(borderline_thickness//2))*2**0.5
    
    

class LinearShapesGenerator(ShapesGenerator):
    """Generate repeated shapes placed along a line with rotation."""
    def __init__(self, shapes_num_range, dist_range, angle_proposals_num, **kwargs):
        """Configure alignment, count, and spacing for linear shapes."""
        super().__init__(**kwargs)

        self.shapes_num_range = shapes_num_range
        self.dist_range = dist_range
        self.proposals_num = angle_proposals_num

    def draw_linear_shapes(self, vecs, shape_size, filled, borderlined, borderline_thickness, max_components, **kwargs):
        """Place multiple rotated copies of a shape along a long segment.

        Returns:
            Tuple compliant with MultishapeMapGenerator expectations.
        """

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
        """Generate one arrangement of linear shapes."""
        return [self.draw_linear_shapes(**self.gen_shape_args(), max_components=max_components)]
    

### SPREADED SHAPES ###

class SpreadedShapesGenerator(ShapesGenerator):
    """Scatter multiple rotated shapes across the canvas with min spacing."""
    def __init__(self, shapes_num_range, min_dist, placement_proposals_num, **kwargs):
        """Configure random placement count, min distance, and proposals."""
        super().__init__(**kwargs)

        self.shapes_num_range = shapes_num_range
        self.min_dist = min_dist
        self.placement_proposals_num = placement_proposals_num

    def __call__(self, max_components, **kwargs):
        """Generate a random set of scattered shapes and masks."""

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
    """Compose multiple randomized patterns into a labeled synthetic map image.

    Produces images, pattern/shape masks, vector endpoints, bboxes, and labels
    required by downstream training pipelines.
    """
    def __init__(self, outputs, size, patterns_num_range, max_components_num, color_rand_range, use_previous_color_prob, min_orig_shape_coverage, grayscale, mask_dtype, bbox_vecs,
                 patterns_prob, line_args, shape_args, line_filled_args, polyline_args, linear_shapes_args, spreaded_shapes_args):
        """Set global generation config and create pattern generators."""

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
            'line_filled': {'generator': LineFilledGenerator(size=size, dtype=mask_dtype, **line_args, **line_filled_args), 'angle_similarity_prevention': True, 'multi_pattern_color_diversity': True, 'color_repetition_filter': ['polyline'], 'single_vec': False, 'type': 'line'},
            'double_line_filled': {'generator': DoubleLineFilledGenerator(size=size, dtype=mask_dtype, **line_args, **line_filled_args), 'angle_similarity_prevention': True, 'multi_pattern_color_diversity': True, 'color_repetition_filter': ['polyline'], 'single_vec': False, 'type': 'line'},
            'polyline': {'generator': PolylineGenerator(size=size, dtype=mask_dtype, **line_args, **polyline_args), 'angle_similarity_prevention': False, 'multi_pattern_color_diversity': False, 'color_repetition_filter': ['double_line_filled', 'line_filled','polyline'], 'single_vec': False, 'type': 'line'},
            'linear_shapes': {'generator': LinearShapesGenerator(size=size, dtype=mask_dtype, **shape_args, **linear_shapes_args), 'angle_similarity_prevention': True, 'multi_pattern_color_diversity': False, 'color_repetition_filter': ['spreaded_shapes'], 'single_vec': True, 'type': 'shape'},
            'spreaded_shapes': {'generator': SpreadedShapesGenerator(size=size, dtype=mask_dtype, **shape_args, **spreaded_shapes_args), 'angle_similarity_prevention': False, 'multi_pattern_color_diversity': True, 'color_repetition_filter': ['linear_shapes', 'spreaded_shapes'], 'single_vec': False, 'type': 'shape'}
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
            'shape_thickness': {'dtype': tf.int32, 'shape': (None,), 'name': 'Oshape_thickness', 'padded_shape': [self.max_components_num]},
            'vecs_pattern_idxs': {'dtype': tf.int32, 'shape': (None,), 'name': 'Pvecs_pattern_idxs', 'padded_shape': [self.max_components_num]},
            'bbox_pattern_idxs': {'dtype': tf.int32, 'shape': (None,), 'name': 'Rbbox_pattern_idxs', 'padded_shape': [self.max_components_num]}
        }

        self.outputs = outputs

        self.outputs_info = dict((k,v) for (k,v) in self.all_outputs_info.items() if k in outputs)

        self.output_dtypes, self.output_shapes, self.output_savenames = [self._get_output_info(name) for name in ['dtype', 'shape', 'name']]

        self.output_padded_shapes = dict(((k, v['padded_shape']) for k, v in self.outputs_info.items()))

    def _get_output_info(self, name):
        """Helper to collect a list of a given property for selected outputs."""
        return [v[name] for (k,v) in self.outputs_info.items()]

    def calc_visible_vec(self, vecs, vec_masks):
        """Estimate representative visible endpoints for each visible vector mask.

        Args:
            vecs: [N, 2, 2] original (y, x) segments.
            vec_masks: [N, H, W, 1] visibility masks after occlusion.
        Returns:
            [N, 2] points closest to each pixel cluster per vector mask.
        """
        dists_to_line = numpy_dist_to_line(self.yx_flat[np.newaxis], vecs)
        vis_vecs = numpy_decode1Dcoords(np.argmax((tf.reduce_sum((vecs[:,:,tf.newaxis]-self.yx_flat[tf.newaxis, tf.newaxis])**2+1e-4, axis=-1)**0.5 \
                                                    -np.transpose(dists_to_line, axes=[0,2,1])**2)*np.reshape(vec_masks, (-1,1,self.size**2)), axis=-1), width=self.size, xy=False)
        
        return vis_vecs
    
    @staticmethod
    def bbox_from_mask(yx, mask):
        """Compute tight bbox quad from a binary mask and coordinate grid."""
        points = yx[mask>0]

        y1, x1 = np.max(points, axis=0)+0.5
        y0, x0 = np.min(points, axis=0)-0.5

        bbox = np.array([[y1, x1],[y0,x1],[y0, x0],[y1, x0]])

        return bbox
    
    def calc_visible_bbox(self, bboxes_masks):
        """Compute visible bboxes for possibly occluded components."""
        bboxes_masks = np.reshape(bboxes_masks, (-1, self.size**2))

        bboxes = np.stack([self.bbox_from_mask(self.yx_flat, bbox_mask) for bbox_mask in bboxes_masks], axis=0)

        return bboxes
        

    def calc_visible_part(self, vecs, bboxes, vecs_masks, borderline_masks, mask, angle, single_vec):
        """Resolve occlusions and derive visible masks, vectors, and bboxes.

        Returns:
            pattern_mask, full_masks, vis_masks, vis_borderline_masks, vis_vecs, vis_bboxes, angle
        """
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
        """Randomly reverse array order to diversify color pairing."""
        return x[::(np.random.randint(0,2)*2-1)]
    
    def gen_colors(self, patterns_num):
        """Generate pairs of foreground/border colors for patterns.

        Ensures alternating palette and optional reuse of previous color.
        """
        if patterns_num%2>0:
            patterns_num += 1

        colors =  np.reshape(np.array([[clr/255 for clr in gen_colors(grayscale=self.grayscale)] for i in range((patterns_num)*2)]), (patterns_num//2, 2, 2, 3))
        #colors = np.concatenate([colors[:1], colors], axis=0)
        previous_color_mask = np.random.binomial(1, self.use_previous_color_prob, (patterns_num//2,1,1,1))

        #colors = colors[1:]*(1-previous_color_mask) + colors[:-1]*previous_color_mask
        #colors = np.array(list(map(self.random_sort, colors)))

        left_colors, right_colors = np.split(colors, 2, axis=1)
        right_colors = right_colors*(1-previous_color_mask) + left_colors*previous_color_mask

        colors = np.reshape(np.concatenate([left_colors, right_colors], axis=1), (patterns_num, 2, 3))
        return colors
    
    def gen_single_color_pair(self):
        """Generate a single pair of colors (shape and borderline)."""
        return np.array([[clr/255 for clr in gen_colors(grayscale=self.grayscale)] for i in range(2)])
    
    @staticmethod
    def get_pattern_drawing(pattern_mask, shape_masks, borderline_masks, shape_color, borderline_color):
        """Blend shape and borderline into an RGB image slice using colors."""
        if borderline_masks is None:
            return pattern_mask*shape_color
        
        shape_mask = np.max(shape_masks, axis=0)
        borderline_masks = np.max(borderline_masks, axis=0)

        shape_mask *= (1-borderline_masks)

        return shape_mask*shape_color + borderline_masks*borderline_color
    
    @staticmethod
    def concat_col(col, empty_shape, dtype):
        """Concatenate list of arrays or return empty typed array if no items."""
        return np.concatenate(col, axis=0) if len(col)>0 else np.zeros(empty_shape, dtype=dtype)
    
    def get_angle_label(self, shape_masks, angle):
        """Average per-pixel angle label over visible masks (masked average)."""
        return np.ma.average(shape_masks*angle[:,np.newaxis, np.newaxis, np.newaxis], weights=shape_masks, axis=0).data
    
    def vec_center_vec(self, vecs, vecs_mask):
        """Compute vector from pixel to nearest point on its assigned line."""
        center_vec = numpy_closest_point_on_line(vecs, self.yx_flat[np.newaxis])-self.yx_flat[np.newaxis]
        return np.ma.average(np.reshape(center_vec, (-1,self.size, self.size,2))*vecs_mask, weights=np.repeat(vecs_mask, 2, -1), axis=0).data
    
    def bbox_center_vec(self, bbox, bbox_mask):
        """Compute vector from pixel to bbox center using masked average."""
        centers = np.mean(bbox, axis=-2, keepdims=True)
        center_vec = centers-self.yx_flat[np.newaxis]
        return np.ma.average(np.reshape(center_vec, (-1,self.size, self.size,2))*bbox_mask, weights=np.repeat(bbox_mask, 2, -1), axis=0).data
    
    @staticmethod
    def get_pattern_idxs_array(vecs, idx):
        """Return an array filled with pattern index for each component."""
        return np.array([idx]*len(vecs), dtype=np.int32)

    def __call__(self, *args):
        """Generate one synthetic sample and all requested outputs.

        Returns:
            List of tensors in the order declared by `outputs` passed to ctor.
        """

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
        vecs_pattern_idxs_col = []
        bbox_pattern_idxs_col = []

        pattern_types = np.random.choice(self.patterns_name, size=patterns_num, replace=True, p=self.patterns_prob)
        colors = self.gen_colors(patterns_num)
        #print(colors)

        background_color = np.array([[[clr/255 for clr in gen_colors(grayscale=self.grayscale)]]])
        previous_pattern = ''

        i=1
        r=0
        pattern_idx = 1
        for pattern_type, color in zip(pattern_types, colors):
            #print(pattern_type)
            generator, angle_similarity_prevention, multi_pattern_color_diversity, color_repetition_filter, single_vec, general_type = self.generator_map[pattern_type].values()
            shape_color, borderline_color = color
            j=0

            for shape_masks, borderline_masks, vecs, bboxes, angle, thickness in generator(max_components=components_left, previous_angles=previous_angles):
                if shape_masks is not None:
                    pattern_mask, full_shape_masks, vis_shape_masks, vis_borderline_masks, vis_vecs, vis_bbox, angle = self.calc_visible_part(vecs, bboxes, shape_masks, borderline_masks, mask, angle, single_vec)
                    if (multi_pattern_color_diversity & (j>0)) | ((previous_pattern in color_repetition_filter) & (r>0)):
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
                            vecs_pattern_idxs_col.append(self.get_pattern_idxs_array(vis_vecs, pattern_idx))

                        if vis_bbox is not None:
                            bbox_masks_col.append(full_shape_masks)
                            bbox_col.append(vis_bbox.astype(np.float32))
                            bbox_pattern_idxs_col.append(self.get_pattern_idxs_array(vis_bbox, pattern_idx))

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
                        
                        pattern_idx += 1
                    else:
                        #print(pattern_type, 'COVERED')
                        None
                    j+=1
                    i+=1

                if (i>self.max_patterns_num) | (components_left<1):
                    break
            
                #if j>0:
                #    pattern_idx += 1
            previous_pattern = pattern_type
            r = abs(r-1)
            if (i>self.max_patterns_num) | (components_left<1):
                break

        img += (1-mask)*background_color

        pattern_masks, shape_masks, vecs_masks, bbox_masks = [self.concat_col(col, (0,self.size, self.size,1), self.mask_dtype) for col in [pattern_masks, shape_masks_col, vecs_masks_col, bbox_masks_col]]
        vecs, bboxes, center_vec = [self.concat_col(vecs_col, (0,2,2), np.float32), self.concat_col(bbox_col, (0,4,2), np.float32), self.concat_col(center_vec_col, (0,self.size, self.size,2), np.float32)]
        vecs_pattern_idxs, bbox_pattern_idxs = [self.concat_col(col, (0,), np.int32) for col in [vecs_pattern_idxs_col, bbox_pattern_idxs_col]]

        shape_thickness = self.concat_col(thickness_col, (0,), np.int32).astype(np.int32)
        thickness_label = np.max(shape_thickness[:, np.newaxis, np.newaxis, np.newaxis]*shape_masks, axis=0).astype(np.int32) if len(shape_thickness)>0 else np.zeros((self.size, self.size, 1), np.int32)

        line_label = np.max(vecs_masks, axis=0) if len(vecs_masks)>0 else np.zeros((self.size, self.size, 1), dtype=self.mask_dtype)
        shape_label = np.max(bbox_masks, axis=0) if len(bbox_masks)>0 else np.zeros((self.size, self.size, 1), dtype=self.mask_dtype)
        center_vec = np.sum(center_vec, axis=0).astype(np.float32) if len(center_vec)>0 else np.zeros((self.size, self.size, 2), dtype=np.float32)

        vecs_mask = np.ones((len(vecs,)), dtype=self.mask_dtype)
        bbox_mask = np.ones((len(bboxes,)), dtype=self.mask_dtype)
        outputs_mapping =  dict(zip(self.all_outputs_info.keys(), [img, angle_label, center_vec, line_label, shape_label, thickness_label, pattern_masks, shape_masks, vecs_masks, bbox_masks, vecs, bboxes, vecs_mask, bbox_mask, shape_thickness, vecs_pattern_idxs, bbox_pattern_idxs]))

        return [outputs_mapping[k] for k in self.outputs]



### dataset preprocessing funcs ###

@tf.function
def colors_random_shift(img, **kwargs):
    """Randomly circular-shift image colors in [0, 1] range per sample."""
    img = (img + tf.random.uniform((3,), 0., 1.)) % 1
    return {'img': img, **kwargs}

def get_mask_weights(x, batch_reg=False):
    """Normalize mask to have mean 1 and optionally normalize over batch.

    Args:
        x: [B,H,W,1] mask.
        batch_reg: if True, normalize total sum to number of pixels.
    Returns:
        Weights tensor with same shape as x.
    """
    w = tf.math.divide_no_nan(x, tf.reduce_mean(flatten(x), axis=-1)[:, tf.newaxis, tf.newaxis, tf.newaxis])
    if batch_reg:
        w *= tf.math.divide_no_nan(tf.cast(tf.reduce_prod(tf.shape(w)), w.dtype), tf.reduce_sum(w))
    return w

def get_shape_class_label(line_label, shape_label, return_all_shapes_mask=False, reversed_label=False):
    """Build 3-channel class label: background, line, shape.

    Args:
        line_label: binary mask of lines.
        shape_label: binary mask of shapes.
        return_all_shapes_mask: optionally return union mask.
        reversed_label: swap order of line/shape channels if True.
    Returns:
        class_label or (class_label, all_shapes_mask).
    """
    i = -1 if reversed_label else 1
    shape_class = tf.concat([line_label, shape_label][::i], axis=-1)
    all_shapes_mask = tf.reduce_max(shape_class, axis=-1, keepdims=True)
    shape_class = tf.cast(tf.concat([1-all_shapes_mask, shape_class], axis=-1), tf.float32)

    all_shapes_mask = tf.cast(all_shapes_mask, tf.float32)

    if return_all_shapes_mask:
        return shape_class, all_shapes_mask
    return shape_class

@tf.function
def op_line_features(img, line_label, shape_label, angle_label, center_vec_label, thickness_label, **kwargs):
    """Package inputs/labels/weights for line feature learning tasks."""
    #shape_class = tf.concat([line_label, shape_label], axis=-1)
    #all_shapes_mask = tf.reduce_max(shape_class, axis=-1, keepdims=True)
    #shape_class = tf.cast(tf.concat([1-all_shapes_mask, shape_class], axis=-1), tf.float32)

    shape_class, all_shapes_mask = get_shape_class_label(line_label, shape_label, return_all_shapes_mask=True)

    line_label = tf.cast(line_label, tf.float32)
    line_label = get_mask_weights(line_label, batch_reg=True) #tf.math.divide_no_nan(line_label, tf.reduce_mean(flatten(line_label), axis=-1)[:, tf.newaxis, tf.newaxis, tf.newaxis])
    #all_shapes_mask = tf.cast(all_shapes_mask, tf.float32)

    #line_label *= tf.math.divide_no_nan(tf.cast(tf.reduce_prod(tf.shape(line_label)), line_label.dtype), tf.reduce_sum(line_label))
    thickness_label = tf.cast(thickness_label, tf.float32)
    all_shapes_mask = get_mask_weights(all_shapes_mask, batch_reg=False) #tf.math.divide_no_nan(all_shapes_mask, tf.reduce_mean(flatten(all_shapes_mask), axis=-1)[:, tf.newaxis, tf.newaxis, tf.newaxis])

    class_mask = tf.ones(tf.shape(shape_class)[:-1], dtype=tf.float32)

    return (img, 
            {'shape_class': shape_class, 'angle': angle_label, 'thickness': thickness_label, 'center_vec': center_vec_label}, 
            {'shape_class': class_mask, 'angle': line_label, 'thickness': all_shapes_mask, 'center_vec': all_shapes_mask}
            )

@tf.function
def blur_img(blur_ratio_range, kernel_size, color_rand_range, img, **kwargs):
    """Randomly blur and jitter colors for augmentation."""
    blur_ratio = tf.random.uniform((), *blur_ratio_range)

    img = tf.clip_by_value(img + tf.random.uniform(tf.shape(img), -color_rand_range, color_rand_range), 0., 1.)
    img = tf.nn.avg_pool2d(img, ksize=kernel_size, strides=1, padding='SAME')*blur_ratio + img*(1-blur_ratio)

    return {'img': img, **kwargs}

@tf.function
def random_flip(**kwargs):
    """Randomly flip tensors horizontally/vertically with 50% chance each."""
    dirs = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)[0]*2-1
    a, b  = dirs[0], dirs[1]
    return dict([(k, v[:,::a, ::b]) for k,v in kwargs.items()])

@tf.function
def op_pixel_similarity(img, pattern_masks, **kwargs):
    """Build per-pixel one-hot similarity label over patterns including background."""
    background_mask = 1 - tf.reduce_sum(pattern_masks, axis=-4, keepdims=True)
    pattern_masks = tf.concat([background_mask, pattern_masks], axis=-4)
    return (img, {'Dot_Similarity': tf.cast(pattern_masks, tf.float32)})

@tf.function
def op_pixel_similarity_shapes(img, pattern_masks, shape_masks, **kwargs):
    """Compute normalized per-pixel similarity over shapes (plus background)."""
    background_mask = 1 - tf.reduce_sum(pattern_masks, axis=-4, keepdims=True)
    label_masks = tf.concat([background_mask, shape_masks], axis=-4)
    label_masks /= tf.reduce_sum(label_masks, axis=-4, keepdims=True)
    return (img, {'Dot_Similarity': tf.cast(label_masks, tf.float32)})

def components_masks_sample_points(vecs_masks, bbox_masks, choosen_components, n, k=1):
    """Sample pixel locations from chosen component masks.

    Returns:
        sample_points: [B, n*k, 2] (y, x) integer points.
        components_masks: gathered masks.
        points_mask: [B, n*k] indicator of valid samples.
    """
    components_masks = tf.concat([vecs_masks, bbox_masks], axis=1)
    components_masks = tf.gather(components_masks, choosen_components, axis=1, batch_dims=1)
    B = tf.shape(choosen_components)[0]
    W = tf.shape(components_masks)[-2]
    flat_components_mask = tf.reshape(components_masks, (B, n, -1))
    points_mask, indices = tf.math.top_k(tf.cast(flat_components_mask, tf.float32)-tf.random.uniform(tf.shape(flat_components_mask), 0.0, 0.1), k=k)
    sample_points = tf.reshape(decode1Dcoords(indices, W)[...,::-1], (-1,n*k,2))
    points_mask = tf.where(points_mask>0., 1., 0.)

    return sample_points, components_masks, points_mask

def vec_sample_point_extraction(vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, n, k=1):
    """Select components, sample points, and assemble vector/bbox labels (split)."""
    components_mask = tf.cast(tf.concat([vecs_mask, bbox_mask], axis=-1), tf.float32)
    choosen_components = tf.math.top_k(components_mask-tf.random.uniform(tf.shape(components_mask), 0.0, 0.1), k=n).indices

    components_mask = tf.cast(tf.gather(components_mask, choosen_components, axis=1, batch_dims=1), tf.float32)

    components_class_mask = tf.cast(tf.gather(tf.concat([tf.ones_like(vecs_mask), tf.zeros_like(bbox_mask)], axis=-1), choosen_components, axis=1, batch_dims=1), tf.float32)

    components_vecs = tf.gather(tf.concat([vecs, bboxes[:,:,0::2]], axis=1), choosen_components, axis=1, batch_dims=1)

    components_class = tf.concat([vecs_mask*2, bbox_mask*1], axis=1)
    components_class = tf.gather(components_class, choosen_components, axis=1, batch_dims=1)

    sample_points, _, points_mask = components_masks_sample_points(vecs_masks, bbox_masks, choosen_components, n, k)
    B = tf.shape(sample_points)[0]
    if k>1:
        components_mask = tf.reshape(points_mask, (B, n*k))
        components_class = tf.repeat(components_class, k, axis=-1)*tf.cast(components_mask, tf.int8)
        components_class_mask = tf.repeat(components_class_mask, k, axis=-1)*components_mask
        components_vecs = tf.repeat(components_vecs, k, axis=1)
        choosen_components = tf.repeat(choosen_components, k, axis=1)

    components_num = tf.reduce_sum(components_mask, axis=None, keepdims=False)
    vecs_weights = components_mask*tf.math.divide_no_nan(tf.cast(n*k*B, tf.float32),components_num) #tf.math.divide_no_nan(components_mask,components_num)

    class_label = tf.cast(tf.one_hot(tf.cast(components_class, tf.int32), 3), tf.float32)
    class_weights = tf.ones((B,n*k), tf.float32)

    vecs_label = (components_class_mask*components_mask)[...,tf.newaxis, tf.newaxis]*components_vecs
    bbox_label = ((1-components_class_mask)*components_mask)[...,tf.newaxis, tf.newaxis]*components_vecs

    mixed_label = tf.concat([vecs_label, bbox_label], axis=-2)

    return choosen_components, sample_points, components_class_mask, mixed_label, class_label, vecs_weights, class_weights

def vec_sample_point_extraction_no_split(vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, n, k=1):
    """Like vec_sample_point_extraction but concatenates vec/bbox label dims."""

    vecs = tf.concat([vecs, vecs], axis=-2)
    bboxes = tf.concat([bboxes[...,0::2,:], bboxes[...,1::2,:]], axis=-2)

    components_mask = tf.cast(tf.concat([vecs_mask, bbox_mask], axis=-1), tf.float32)
    choosen_components = tf.math.top_k(components_mask-tf.random.uniform(tf.shape(components_mask), 0.0, 0.1), k=n).indices

    components_mask = tf.cast(tf.gather(components_mask, choosen_components, axis=1, batch_dims=1), tf.float32)

    vecs_label = tf.gather(tf.concat([vecs, bboxes], axis=1), choosen_components, axis=1, batch_dims=1)

    components_class = tf.concat([vecs_mask*2, bbox_mask*1], axis=1)
    components_class = tf.gather(components_class, choosen_components, axis=1, batch_dims=1)

    sample_points, _, points_mask = components_masks_sample_points(vecs_masks, bbox_masks, choosen_components, n, k)
    B = tf.shape(sample_points)[0]
    if k>1:
        components_mask = tf.reshape(points_mask, (B, n*k))
        components_class = tf.repeat(components_class, k, axis=-1)*tf.cast(components_mask, tf.int8)
        vecs_label = tf.repeat(vecs_label, k, axis=1)*components_mask[...,tf.newaxis, tf.newaxis]
        choosen_components = tf.repeat(choosen_components, k, axis=1)

    components_num = tf.reduce_sum(components_mask, axis=None, keepdims=False)
    vecs_weights = components_mask*tf.math.divide_no_nan(tf.cast(n*k*B, tf.float32),components_num) #tf.math.divide_no_nan(components_mask,components_num)

    class_label = tf.cast(tf.one_hot(tf.cast(components_class, tf.int32), 3), tf.float32)
    class_weights = tf.ones((B,n*k), tf.float32)


    return choosen_components, sample_points, vecs_label, class_label, vecs_weights, class_weights

@tf.function
def op_k_shape_points_vecs(img, vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, n, k=1, shape_thickness=None, vec_label=True, add_thickness=False, add_probe_label=False, **kwargs):
    """Dataset op: sample k points on n components and build labels/weights."""
    choosen_components, sample_points, vecs_label, class_label, vecs_weights, class_weights = \
        vec_sample_point_extraction_no_split(vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, n, k=k)
    
    inputs = {'img': img, 'sample_points': tf.cast(sample_points, tf.float32)}
    if vec_label:
        labels = {'vecs': vecs_label, 'class': class_label}
        weights = {'vecs': vecs_weights, 'class': class_weights}
    else:
        labels = {}
        weights = {}

    if add_thickness | add_probe_label:
        thickness_label = tf.cast(get_indexed_vec_thickness_label(shape_thickness, choosen_components), tf.float32)

    if add_thickness:
        labels['thickness'] = thickness_label
        weights['thickness'] = tf.expand_dims(vecs_weights, axis=-1)

    if add_probe_label:
        labels['probe'] = tf.concat([tf.reshape(vecs_label, (-1,n*k,8)), thickness_label], axis=-1)
        weights['probe'] = vecs_weights

    return inputs, labels, weights

@tf.function
def op_sample_points_vecs(img, vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, n, **kwargs):
    """Dataset op: sample one point per component and split vec/bbox labels."""
    
    _, sample_points, components_class_mask, mixed_label, class_label, vecs_weights, class_weights = \
        vec_sample_point_extraction(vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, n)
    #class_split = tf.stack([components_class, 1-components_class], axis=-1)

    return ({'img': img, 'sample_points': sample_points, 'class_split': components_class_mask}, 
            {'vecs': mixed_label, 'class': class_label}, 
            {'vecs': vecs_weights, 'class': class_weights})

def get_indexed_vec_thickness_label(thickness, idxs):
    """Gather thickness labels by sampled component indices (handles vec/bbox)."""
    return tf.expand_dims(tf.gather(tf.concat([thickness, thickness], axis=1), idxs, axis=1, batch_dims=1), axis=-1)

@tf.function
def op_sample_points_vecs_with_thickness(img, vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, shape_thickness, n, **kwargs):
    """Dataset op: sample components and return vecs/class/thickness triplet."""
    
    choosen_components, sample_points, components_class_mask, mixed_label, class_label, vecs_weights, class_weights = \
        vec_sample_point_extraction(vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, n)
    #class_split = tf.stack([components_class, 1-components_class], axis=-1)

    thickness_label = get_indexed_vec_thickness_label(shape_thickness, choosen_components)

    return {'inputs': {'img': img, 'sample_points': sample_points, 'class_split': components_class_mask}, 
            'labels': {'vecs': mixed_label, 'class': class_label, 'thickness': tf.cast(thickness_label, tf.float32)}, 
            'weights': {'vecs': vecs_weights, 'class': class_weights, 'thickness': tf.expand_dims(vecs_weights, axis=-1)}}

@tf.function
def op_dict_free_pass(inputs, labels, weights):
    """Identity op for pipelines expecting a mapping transformation."""
    return (inputs, labels, weights)

'''def random_vec_angles(vecs, vecs_mask, angle_samples_num):
    vecs_mask = tf.cast(vecs_mask, tf.float32)
    vec_angles = calc_2x2_vec_angle(vecs)
    #vec_angles = tf.where(vec_angles<0, vec_angles+math.pi, vec_angles)

    initial_idx = tf.math.top_k(vecs_mask+tf.random.uniform(tf.shape(vecs_mask), 0.0, 0.1), k=1).indices
    angle_input = tf.gather(vec_angles, initial_idx, axis=1, batch_dims=1)
    angle_input_mask = tf.gather(vecs_mask, initial_idx, axis=1, batch_dims=1)
    expanded_angles = tf.expand_dims(vec_angles, axis=1)
    angle_mask = tf.expand_dims(vecs_mask, axis=1)

    for _ in range(angle_samples_num-1):
        diffs = tf.reduce_min(two_side_angle_diff(tf.expand_dims(angle_input, axis=2), expanded_angles)*angle_mask, axis=1)
        idx = tf.argmax(diffs, axis=-1)[:,tf.newaxis]

        angle_input = tf.concat([angle_input, tf.gather(vec_angles, idx, axis=1, batch_dims=1)], axis=-1)
        angle_input_mask = tf.concat([angle_input_mask, tf.gather(vecs_mask, idx, axis=1, batch_dims=1)], axis=-1)

    cutted_vecs_mask = vecs_mask[:,:angle_samples_num]
    angle_input = angle_input*cutted_vecs_mask + tf.random.uniform(tf.shape(angle_input),-math.pi, math.pi)*(1-cutted_vecs_mask)

    return angle_input'''

def random_vec_angles(vecs, vecs_mask, angle_samples_num, random_angle_weight=0.1):
    """Sample diverse reference angles from existing vectors with fallback noise."""
    vecs_mask = tf.cast(vecs_mask, tf.float32)
    vec_angles = calc_2x2_vec_angle(vecs)
    #vec_angles = tf.where(vec_angles<0, vec_angles+math.pi, vec_angles)
    vec_angles = vecs_mask*vec_angles + (1-vecs_mask)*tf.random.uniform(tf.shape(vec_angles),-math.pi, math.pi)

    initial_idx = tf.math.top_k(vecs_mask+tf.random.uniform(tf.shape(vecs_mask), 0.0, 0.1), k=1).indices
    angle_input = tf.gather(vec_angles, initial_idx, axis=1, batch_dims=1)

    expanded_angles = tf.expand_dims(vec_angles, axis=1)
    angle_mask = tf.expand_dims(tf.where(vecs_mask==0., random_angle_weight, vecs_mask), axis=1)

    for _ in range(angle_samples_num-1):
        diffs = tf.reduce_min(two_side_angle_diff(tf.expand_dims(angle_input, axis=2), expanded_angles)*angle_mask, axis=1)
        idx = tf.argmax(diffs, axis=-1)[:,tf.newaxis]

        angle_input = tf.concat([angle_input, tf.gather(vec_angles, idx, axis=1, batch_dims=1)], axis=-1)

    return angle_input

def vec_label_prep(vecs, bboxes):
    """Prepare concatenated vec and bbox endpoints for downstream labels."""
    vecs = tf.concat([vecs, vecs], axis=-2)
    bboxes = tf.concat([bboxes[...,0::2,:], bboxes[...,1::2,:]], axis=-2)

    return vecs, bboxes


def prepare_shapes_label(vecs, bboxes, vecs_mask, bbox_mask, n):
    """Select up to n components and assemble labels/weights for training."""

    vecs, bboxes = vec_label_prep(vecs, bboxes)

    components_mask = tf.cast(tf.concat([vecs_mask, bbox_mask], axis=-1), tf.float32)
    choosen_components = tf.math.top_k(components_mask+tf.random.uniform(tf.shape(components_mask), 0.0, 0.1), k=n).indices

    components_mask = tf.cast(tf.gather(components_mask, choosen_components, axis=1, batch_dims=1), tf.float32)
    vecs_label = tf.gather(tf.concat([vecs, bboxes], axis=1), choosen_components, axis=1, batch_dims=1)

    components_class = tf.concat([vecs_mask*2, bbox_mask*1], axis=1)
    components_class = tf.gather(components_class, choosen_components, axis=1, batch_dims=1)*tf.cast(components_mask, tf.int8)

    B = tf.shape(choosen_components)[0]
    components_num = tf.reduce_sum(components_mask, axis=None, keepdims=True)
    vecs_weights = components_mask*tf.math.divide_no_nan(tf.cast(n*B, tf.float32),components_num)

    class_label = tf.cast(tf.one_hot(components_class, 3), tf.float32)
    class_weights = tf.ones((B,n), tf.float32)

    return vecs_label, class_label, vecs_weights, class_weights, components_mask, choosen_components

@tf.function
def op_rotated_enc(img, vecs, bboxes, vecs_mask, bbox_mask, angle_samples_num, max_components_num):
    """Dataset op: angle sampling plus vector labels for encoder training."""

    angle_input = random_vec_angles(vecs, vecs_mask, angle_samples_num)

    vecs_label, class_label, vecs_weights, class_weights, vecs_mask, _ = prepare_shapes_label(vecs, bboxes, vecs_mask, bbox_mask, max_components_num)

    return ({'img': img, 'angle_input': angle_input},
            {'vecs': vecs_label, 'class': class_label, 'vecs_weights': vecs_weights, 'vecs_mask': vecs_mask})

'''
Deprecated due to vec label multiplication on polyline corners

def vecs_full_label(vecs, bboxes, vecs_masks, bbox_masks):
    vecs, bboxes = vec_label_prep(vecs, bboxes)

    vec_components = tf.concat([vecs, bboxes], axis=1)
    components_masks = tf.cast(tf.concat([vecs_masks, bbox_masks], axis=1), tf.float32)
    vec_label = tf.reduce_sum(vec_components[...,tf.newaxis, tf.newaxis,:,:]*components_masks[...,tf.newaxis], axis=-5)

    return vec_label


def vec_rotated_full_label(vecs, bboxes, vecs_masks, bbox_masks, angle_input):
    vecs, bboxes = vec_label_prep(vecs, bboxes) # [B, N, 4, 2]

    angle_samples_num = tf.shape(angle_input)[-1] # [B, A]
    # bbox rotation
    rot_matrix = gen_rot_matrix_yx(-angle_input) # [B, A, 2, 2]
    bbox_center = tf.reduce_mean(bboxes, axis=-2, keepdims=True) # [B, N, 1, 2]
    bboxes = tf.matmul(tf.expand_dims(bboxes-bbox_center, axis=-3), tf.expand_dims(rot_matrix, axis=-4)) + tf.expand_dims(bbox_center, axis=-3) # [B, N, A, 4, 2]
    # vec repeat
    vecs = tf.repeat(tf.expand_dims(vecs, axis=-3), angle_samples_num, axis=-3) # [B, N, A, 4, 2]

    vec_components = tf.concat([vecs, bboxes], axis=1) # [B, 2N, A, 4, 2]
    components_masks = tf.cast(tf.concat([vecs_masks, bbox_masks], axis=1), tf.float32) # [B, 2N, H, W, 1]
    vec_label = tf.reduce_sum(vec_components[...,tf.newaxis, tf.newaxis,:,:,:]*components_masks[...,tf.newaxis, tf.newaxis], axis=1) # [B, H, W, A, 4, 2]

    return vec_label'''

def vec_rotated_full_label(vecs, bboxes, vecs_masks, bbox_masks, line_label, shape_label, angle_input, vec_rotation=True):
    """Assemble per-pixel vector labels combining lines and a single bbox.

    Optionally rotate the bbox around its center according to angle_input.
    """

    vecs_masks_sums = tf.reduce_sum(vecs_masks, axis=1)
    vecs_masks_t = tf.transpose(vecs_masks[...,0], [0,2,3,1])
    top_vecs_idxs = tf.math.top_k(vecs_masks_t, k=2).indices
    top_vecs_idx = top_vecs_idxs[...,:1]

    corner_mask = tf.expand_dims(tf.cast(tf.where(vecs_masks_sums > 1, 1, 0), tf.float32), axis=-1)

    single_vecs = tf.squeeze(tf.gather(vecs, top_vecs_idx, batch_dims=1), axis=-3)
    corner_vecs = tf.gather(vecs, top_vecs_idxs, batch_dims=1)

    single_vecs = tf.concat([single_vecs, single_vecs], axis=-2)
    corner_vecs = tf.squeeze(tf.concat(tf.split(corner_vecs, 2, axis=-3), axis=-2), axis=-3)

    vec_joint_label = ((1-corner_mask)*single_vecs + corner_mask*corner_vecs)*tf.cast(line_label[...,tf.newaxis], tf.float32)

    bboxes_label = tf.concat([bboxes[...,0::2,:], bboxes[...,1::2,:]], axis=-2)
    bbox_masks_t = tf.transpose(bbox_masks[...,0], [0,2,3,1])
    bbox_idx = tf.argmax(bbox_masks_t, axis=-1)[...,tf.newaxis]
    single_bbox = tf.squeeze(tf.gather(bboxes_label, bbox_idx, batch_dims=1), axis=-3)*tf.cast(shape_label[...,tf.newaxis], tf.float32)

    if vec_rotation:
        angle_samples_num = tf.shape(angle_input)[-1]
        rot_matrix = gen_rot_matrix_yx(angle_input)

        vec_joint_label = tf.repeat(tf.expand_dims(vec_joint_label, axis=-3), angle_samples_num, axis=-3)

        bbox_center = tf.reduce_mean(single_bbox, axis=-2, keepdims=True)  
        single_bbox = tf.matmul(tf.expand_dims(single_bbox-bbox_center, axis=-3), rot_matrix[:,tf.newaxis, tf.newaxis]) + tf.expand_dims(bbox_center, axis=-3)

    full_vec_label = vec_joint_label + single_bbox

    return full_vec_label

def vec_angle_filter(vecs, angle_input, vecs_mask, vecs_masks, line_label, radian_range):
    """Filter vectors not matching sampled angles; return filtered masks/labels."""
    vec_angles = calc_2x2_vec_angle(vecs)
    angle_diff = tf.abs(vec_angles[...,tf.newaxis] - angle_input[...,tf.newaxis, :])
    filtered_vecs_mask = tf.cast(tf.reduce_min(angle_diff, axis=-1)<=radian_range, vecs_masks.dtype)*vecs_mask
    filtered_vecs_masks = filtered_vecs_mask[...,tf.newaxis, tf.newaxis, tf.newaxis]*vecs_masks
    filtered_line_label = tf.reduce_max(filtered_vecs_masks, axis=1, keepdims=False)
    filtered_line_anti_mask = (1-filtered_line_label)*line_label
    return filtered_vecs_mask, filtered_vecs_masks, filtered_line_label, filtered_line_anti_mask

@tf.function
def op_rotated_enc_full_label(img, vecs, bboxes, vecs_mask, bbox_mask, vecs_masks, bbox_masks, line_label, shape_label, thickness_label, angle_samples_num, random_angle_weight, rotated_bbox_label, angle_input_rand_range, splitted_conf=False):
    """Dataset op: build rotated full vector labels with class/thickness targets."""

    angle_input = random_vec_angles(vecs, vecs_mask, angle_samples_num, random_angle_weight=random_angle_weight)

    radian_range = angle_input_rand_range/180*math.pi
    if angle_input_rand_range>0.: 
        angle_input += tf.random.uniform(tf.shape(angle_input), -radian_range, radian_range)

    shape_class, all_shapes_mask = get_shape_class_label(line_label, shape_label, return_all_shapes_mask=True, reversed_label=True)

    #filter vecs that are too far from the angle_input
    vecs_mask, vecs_masks, line_label, filtered_line_anti_mask = vec_angle_filter(vecs, angle_input, vecs_mask, vecs_masks, line_label, radian_range)
    class_weights = tf.squeeze(get_mask_weights(tf.cast(1-filtered_line_anti_mask, tf.float32)), axis=-1)

    '''if rotated_bbox_label:
        vec_label = vec_rotated_full_label(vecs, bboxes, vecs_masks, bbox_masks, angle_input)
    else:
        vec_label = vecs_full_label(vecs, bboxes, vecs_masks, bbox_masks)'''
    
    vec_label = vec_rotated_full_label(vecs, bboxes, vecs_masks, bbox_masks, line_label, shape_label, angle_input)
    if splitted_conf:
        vec_label = tf.squeeze(vec_label, axis=-3)
    
    filtered_all_shapes_mask = all_shapes_mask - tf.cast(filtered_line_anti_mask, tf.float32)
    vec_weights = get_mask_weights(filtered_all_shapes_mask, batch_reg=False)

    thickness_label = tf.cast(thickness_label, tf.float32)*tf.cast(line_label, tf.float32)
    thickness_weights = tf.squeeze(get_mask_weights(tf.cast(line_label, tf.float32), batch_reg=True), axis=-1)

    inputs = {'img': img, 'angle_input': angle_input}
    labels = {'vecs': vec_label, 'class': shape_class, 'thickness': thickness_label}
    weights = {'vecs': vec_weights, 'class': class_weights, 'thickness': thickness_weights}

    if splitted_conf:
        labels['conf'] = filtered_all_shapes_mask
        weights['conf'] = tf.squeeze(get_mask_weights(all_shapes_mask, batch_reg=False), axis=-1)

    return (inputs, labels, weights)


@tf.function
def op_all_sample_points_vecs_with_thickness(img, vecs_mask, bbox_mask, vecs_masks, bbox_masks, vecs, bboxes, shape_thickness, max_components_num, **kwargs):
    """Dataset op: sample all components and return vecs/class/thickness with points."""
    
    vecs_label, class_label, vecs_weights, class_weights, vecs_mask, choosen_components = prepare_shapes_label(vecs, bboxes, vecs_mask, bbox_mask, max_components_num)

    thickness_label = get_indexed_vec_thickness_label(shape_thickness, choosen_components)

    sample_points, _, _ = components_masks_sample_points(vecs_masks, bbox_masks, choosen_components, max_components_num)

    return ({'img': img, 'sample_points': sample_points}, 
            {'vecs': vecs_label, 'class': class_label, 'thickness': tf.cast(thickness_label, tf.float32)}, 
            {'vecs': vecs_weights, 'class': class_weights, 'thickness': tf.expand_dims(vecs_weights, axis=-1)})

@tf.function
def op_freq_space_angle_mask(img, vecs, vecs_mask, size, threshold, binarise, **kwargs):
    """Compute angle-consistency mask in frequency space for supervision."""
    vecs_angles = calc_2x2_vec_angle(vecs)
    angles_map = fft_angles(size)

    angle_scores = tf.reduce_max((1-two_side_angle_diff(angles_map[tf.newaxis], vecs_angles[:,tf.newaxis, tf.newaxis]))*tf.cast(vecs_mask, tf.float32)[:,tf.newaxis, tf.newaxis], axis=-1, keepdims=True)
    angle_scores = tf.where(angle_scores>threshold, 1. if binarise else angle_scores, 0.)

    return ({'img': img}, {'angle_mask': angle_scores})

@tf.function
def op_image_autoencoder(img, size, center_shift_range):
    """Return single-channel image and center points for autoencoder tasks."""
    B = tf.shape(img)[0]
    random_channel = tf.random.uniform((B,1), 0, 3, dtype=tf.int32)
    single_channel_img = tf.transpose(tf.gather(tf.transpose(img, [0,3,1,2]), random_channel, axis=1, batch_dims=1), [0,2,3,1])
    center_point = (size-1)/2

    center_points = center_point + tf.random.uniform((B,2), -center_shift_range, center_shift_range)

    return ({'img': single_channel_img, 'center_points': center_points}, {'target_img': single_channel_img})

### DATASET GENERATOR ###


class DatasetGenerator:
    """Utility to generate, serialize, and load TF datasets for vector maps."""
    def __init__(self, map_generator, ds_path, fold_size, parallel_calls, padded_batch, output_filter, preprocess_funcs, set_shapes=False, **kwargs):
        """Initialize dataset generator with pipeline and storage settings."""

        self.fmg = map_generator

        self.ds_path = ds_path
        self.fold_size = fold_size
        self.parallel_calls = parallel_calls
        self.padded_batch = padded_batch

        self.preprocess_funcs = preprocess_funcs

        self.storage_client = storage.Client()

        self.output_filter = output_filter
        self.padded_shapes = dict([(elem, self.fmg.output_padded_shapes[elem]) for elem in output_filter]) if output_filter is not None else self.fmg.output_padded_shapes

        self.set_shapes = set_shapes

    @tf.function
    def _filter_outputs(self, inputs):
        """Filter a mapping to include only requested outputs."""
        return dict([(elem, inputs[elem]) for elem in self.output_filter])

    @tf.function
    def _gen_images(self, *args):
        """Generate a single sample by invoking the map generator via py_function."""

        inputs = tf.py_function(self.fmg, [], self.fmg.output_padded_shapes)

        return inputs
    
    @tf.function
    def _set_shapes(self, *args):
        """Attach static shapes to tensors to help graph building and tracing."""
        # shapes definition
        inputs = args

        for input, input_shape in zip(inputs, self.fmg.output_shapes):
            input.set_shape(input_shape)

        return inputs
    
    def _gen_feature_description(self):
        """Schema mapping for TFRecord parsing (all fields are serialized tensors)."""
        return {name: tf.io.FixedLenFeature([], tf.string) for name in self.fmg.output_savenames}
    
    @staticmethod
    def create_path_if_needed(path):
        """Create directory hierarchy if it does not exist."""
        folders = os.path.split(path)
        curr_path = ''
        for folder in folders:
            curr_path = os.path.join(curr_path, folder)
            if not os.path.exists(curr_path):
                os.mkdir(curr_path)

    @staticmethod
    def _bytes_feature(value):
        """Wrap a tensor buffer as a TF Example Feature (bytes_list)."""
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_example(self, names, inputs):
        """Serialize named tensors into a single tf.train.Example."""
        feature = {name: self._bytes_feature(tf.io.serialize_tensor(x)) for name, x in zip(names, inputs)}

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    @staticmethod
    def _parse_function(example_proto, dtypes, feature_description):
        """Parse and deserialize a single Example into typed tensors."""
        inputs = tf.io.parse_single_example(example_proto, feature_description).values()

        inputs = [tf.io.parse_tensor(x, x_type) for x, x_type in zip(inputs, dtypes)]

        return inputs
    
    def save_tfrec_dataset(self, folds_num=1, starting_num=0):
        """Generate and save multiple TFRecord files to ds_path."""
        self.create_path_if_needed(self.ds_path)
        names = self.fmg.output_savenames
        for fold in range(folds_num):
            ds = self._new_dataset(parallel_calls=1)

            #self.ds = self.ds.map(tf_serialize_example, self.cfg.num_parallel_calls)
            print(f'\n\033[1msaving fold {fold+1}/{folds_num}\033[0m')
            pb = tf.keras.utils.Progbar(self.fold_size)
            with tf.io.TFRecordWriter(f"{self.ds_path}/ds-{self.ds_path.split(os.sep)[-1] if False else ''}{fold+starting_num}.tfrec") as writer:
                for inputs in ds:
                    writer.write(self.serialize_example(names,inputs))
                    pb.add(1)

    def upload_dataset_to_storage(self, name):
        """Upload all files from ds_path to a GCS bucket named project+name."""
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
        """Download all blobs from a GCS bucket named project+name into ds_path."""
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
        """Delete the GCS bucket named project+name (force=True)."""
        bucket_name = self.storage_client.project + name
        self.storage_client.get_bucket(bucket_name).delete(force=True)

    def dataset_speed_test(self,test_iters, batch_size, ds_iter):
        """Benchmark iteration speed over a dataset iterator."""
        print('\n\033[1mDataset generator speed test\033[0m')
        start_time = time.time()
        for _ in range(test_iters):
            _ = next(ds_iter)
        proc_time = time.time()-start_time

        print('time per batch: %.3fs | time per example: %.3fs' % (proc_time/test_iters,proc_time/(batch_size*test_iters)))

    @staticmethod
    def delete_dataset(path):
        """Remove dataset directory recursively using shell rm -r."""
        os.system(f'rm -r {path}')

    def get_ds_sizes(self):
        """List TFRecord file sizes in ds_path in MB as strings."""
        return ['{}: {:.3f} MB'.format(filename, os.path.getsize(os.path.join(self.ds_path, filename))*1e-6) for filename in os.listdir(self.ds_path)]

    @tf.function
    def _tf_map_drawing(self, *args):
        """TF graph wrapper to call numpy-based map generator."""
        return tf.numpy_function(self.fmg, [], self.fmg.output_dtypes)

    def _new_dataset(self, parallel_calls):
        """Create a tf.data.Dataset that generates examples on the fly."""
        ds = tf.data.Dataset.range(self.fold_size)
        ds = ds.map(self._tf_map_drawing, num_parallel_calls=parallel_calls, deterministic=False)

        return ds
    
    def _load_dataset(self, val_idxs, validation):
        """Load TFRecord dataset files either for training or validation split."""
        feature_description = self._gen_feature_description()
        ds_files = [os.path.join(self.ds_path, filename) for i, filename in enumerate(os.listdir(self.ds_path)) if (i in val_idxs if validation else i not in val_idxs)]
        ds = tf.data.TFRecordDataset(ds_files, num_parallel_reads=self.parallel_calls, buffer_size=int(50*1e6))

        ds = ds.map(lambda x: self._parse_function(x, self.fmg.output_dtypes, feature_description), num_parallel_calls=self.parallel_calls)

        return ds, len(ds_files)*self.fold_size
    
    def _map_names(self, *args):
        """Map list of tensors to a dict keyed by output names."""
        return dict(zip(self.fmg.outputs, args))

    def dataset(self, batch_size=0, repeat=True, from_saved=False, validation=False, val_idxs=[], shuffle_buffer_size=0):
        """Build a batched tf.data pipeline with optional preprocessing.

        Args:
            batch_size: batch size or 0 for unbatched.
            repeat: if True, repeat indefinitely.
            from_saved: if True, read TFRecords instead of generating.
            validation: if True, use validation split indices.
            val_idxs: list of fold indices to use as validation.
            shuffle_buffer_size: shuffle buffer for training set.
        Returns:
            (dataset, steps) where steps is number of batches per epoch.
        """
        
        if not from_saved:
            ds = self._new_dataset(self.parallel_calls)
            records = self.fold_size
        else:
            ds, records = self._load_dataset(val_idxs, validation)

        if self.set_shapes:
            ds = ds.map(self._set_shapes, num_parallel_calls=self.parallel_calls)
            
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
            ds = ds.prefetch(2)
        if repeat:
            ds = ds.repeat()

        return ds, steps

