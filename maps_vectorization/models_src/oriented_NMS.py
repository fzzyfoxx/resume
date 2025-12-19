import tensorflow as tf
import numpy as np
from models_src.VecModels import gen_rot_matrix_yx
from models_src.fft_lib import yx_coords
from models_src.Attn_variations import SqueezeImg

def line_intersection(p1, p2, p3, p4):
    """Compute the intersection point of two infinite lines defined by point pairs.

    Uses the analytic 2D line-line intersection formula in a fully vectorized way.
    Lines are defined by (p1, p2) and (p3, p4). This does not check segment bounds;
    downstream code filters intersections that lie inside rectangles.

    Args:
        p1: Tensor of shape (..., 2). First point on line 1.
        p2: Tensor of shape (..., 2). Second point on line 1.
        p3: Tensor of shape (..., 2). First point on line 2.
        p4: Tensor of shape (..., 2). Second point on line 2.

    Returns:
        Tensor of shape (..., 2) with the intersection coordinates.
    """
    x1, y1 = p1[..., 0], p1[..., 1]
    x2, y2 = p2[..., 0], p2[..., 1]
    x3, y3 = p3[..., 0], p3[..., 1]
    x4, y4 = p4[..., 0], p4[..., 1]

    Px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))

    return tf.stack([Px, Py], axis=-1)

def is_point_in_rect(point, rect, epsilon=1e-6):
    """Point-in-rotated-rectangle test via dot-product projections.

    Treats the rectangle as two edges AB and BC. A point M is inside iff the
    projections of AM onto AB and BM onto BC both lie within [0, |AB|] and [0, |BC|].

    Args:
        point: Tensor of shape (..., P, 2) or (..., 2). Points to test.
        rect: Tensor of shape (..., 4, 2). Rectangle corners in order.
        epsilon: Float tolerance added to handle numeric instability on edges.

    Returns:
        Boolean tensor of shape (..., P) or (...) indicating inclusion.
    """
    AB = rect[..., 1, tf.newaxis, :] - rect[..., 0, tf.newaxis, :]
    AM = point - rect[..., 0, tf.newaxis, :]
    BC = rect[..., 2, tf.newaxis, :] - rect[..., 1, tf.newaxis, :]
    BM = point - rect[..., 1, tf.newaxis, :]
    ABAM = tf.reduce_sum(AB * AM, axis=-1)
    BCBM = tf.reduce_sum(BC * BM, axis=-1)
    ABAB = tf.reduce_sum(AB * AB, axis=-1)
    BCBC = tf.reduce_sum(BC * BC, axis=-1)

    return (0<=ABAM+epsilon) & (ABAM<=ABAB+epsilon) & (0<=BCBM+epsilon) & (BCBM<=BCBC+epsilon)

def rotate_rectangle(rect):
    """Randomly rotate a rectangle around its center.

    Args:
        rect: Tensor of shape (..., 4, 2). Rectangle vertices.

    Returns:
        Tuple (rect_rot, angle) where rect_rot is rotated rect with same shape
        and angle is the sampled rotation in radians.
    """
    angle = tf.random.uniform([], minval=0, maxval=2*np.pi)
    rot_matrix = gen_rot_matrix_yx(angle)
    rect_center = tf.reduce_mean(rect, axis=-2, keepdims=True)
    rect_rot = tf.matmul(rect - rect_center, rot_matrix) + rect_center
    return rect_rot, angle

def iterative_intersection_points(rect1, rect2):
    """Compute all pairwise line intersections between rectangle edges (looped).

    Iterates over the 4 edges of rect1 and rect2 to compute 16 line-line
    intersections. Segment bounds are not enforced here.

    Args:
        rect1: Tensor of shape (..., 4, 2).
        rect2: Tensor of shape (..., 4, 2).

    Returns:
        Tensor of shape (..., 16, 2) with all intersections.
    """
    # Find all intersection points
    points = []
    for i in range(4):
        for j in range(4):
            p1, p2 = rect1[..., i, :], rect1[..., (i + 1) % 4, :]
            p3, p4 = rect2[..., j, :], rect2[..., (j + 1) % 4, :]

            intersect = line_intersection(p1, p2, p3, p4)
            points.append(intersect)
    points = tf.stack(points, axis=-2)
    return points

def parallel_intersection_points(rect1, rect2):
    """Vectorized all-edges intersection between two rectangles.

    Forms a sliding window of edges for both rectangles and computes all 16
    line intersections in parallel, then reshapes to (..., 16, 2).

    Args:
        rect1: Tensor of shape (..., 4, 2).
        rect2: Tensor of shape (..., 4, 2).

    Returns:
        Tensor of shape (..., 16, 2) with intersections for each edge pair.
    """
    rect1_exp = tf.expand_dims(tf.concat([rect1, rect1[..., :1, :]], axis=-2), axis=-2)
    rect2_exp = tf.expand_dims(tf.concat([rect2, rect2[..., :1, :]], axis=-2), axis=-3)

    p1, p2 = rect1_exp[...,:-1,:,:], rect1_exp[...,1:,:,:]
    p3, p4 = rect2_exp[...,:-1,:], rect2_exp[...,1:,:]

    points_exp = line_intersection(p1, p2, p3, p4)
    batch_shape = points_exp.shape[:-3]
    points = tf.reshape(points_exp, batch_shape+(16, 2))

    return points

def random_roll(rect, max_roll=4, axis=-2):
    """Randomly roll vertices along an axis to change starting corner.

    Useful to canonicalize or augment polygon vertex ordering.

    Args:
        rect: Tensor (..., 4, 2).
        max_roll: Maximum absolute shift.
        axis: Axis along which to roll (default -2 for vertex dimension).

    Returns:
        Tensor with same shape as input, rolled along the given axis.
    """
    roll = tf.random.uniform([], minval=-max_roll, maxval=max_roll, dtype=tf.int32)
    rect = tf.roll(rect, shift=roll, axis=axis)
    return rect

def points_duplicates_mask(points, epsilon=1e-2):
    """Compute a mask that filters out duplicate points within a tolerance.

    Builds a pairwise distance grid and suppresses symmetric duplicates using
    a triangular mask so each coincident cluster contributes one representative.

    Args:
        points: Tensor of shape (..., N, 2).
        epsilon: Distance threshold below which points are considered equal.

    Returns:
        Tensor of shape (..., N) with 1 for unique representatives and 0 for duplicates.
    """
    pts_num = points.shape[-2]
    yx = yx_coords((pts_num, pts_num))
    duplicate_diag_mask = tf.cast(yx[...,0]>yx[...,1], tf.int8)

    points_grid = tf.cast(tf.reduce_sum(tf.abs(tf.expand_dims(points, axis=-2)-tf.expand_dims(points, axis=-3)), axis=-1)<epsilon, tf.int8)
    duplicate_filter_mask = tf.reduce_max(points_grid * duplicate_diag_mask, axis=-1)

    return 1-duplicate_filter_mask

def triangle_area(p1, p2, p3):
    """Area of a triangle via the shoelace formula.

    Args:
        p1: Tensor (..., 2). First vertex (y, x).
        p2: Tensor (..., 2). Second vertex (y, x).
        p3: Tensor (..., 2). Third vertex (y, x).

    Returns:
        Tensor (..., 1) area magnitude of the triangle fan elements.
    """
    y1, x1 = tf.split(p1, 2, axis=-1)
    y2, x2 = tf.split(p2, 2, axis=-1)
    y3, x3 = tf.split(p3, 2, axis=-1)
    
    area = 0.5 * tf.abs(y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2))
    return area

def convex_polygon_area(vertices, vertices_mask, batch_dims=1):
    """Area of a (possibly sparse) convex polygon via fan triangulation.

    Reorders masked vertices by polar angle around the centroid and accumulates
    triangle areas of a fan starting from the first (origin) vertex. Non-selected
    positions are replaced by the origin to keep shapes consistent.

    Args:
        vertices: Tensor (..., N, 2). Polygon vertices (y, x).
        vertices_mask: Tensor (..., N). 1 if vertex is present, else 0.
        batch_dims: Number of batch dimensions to preserve for tf.gather.

    Returns:
        Tensor (...) with polygon area.
    """
    origin_point = vertices[...,:1, :]
    vertices_mask = tf.cast(vertices_mask, vertices.dtype)[...,tf.newaxis]
    proper_points = vertices_mask * vertices + (1-vertices_mask) * origin_point

    # Calculate vertices angles against centroid and shift them so that origin point is at 0 angle
    # Convert [-pi, pi] to [0, 2*pi]
    centroid = tf.reduce_sum(proper_points*vertices_mask, axis=-2, keepdims=True) / (tf.reduce_sum(vertices_mask, axis=-2, keepdims=True)+1e-6)
    proper_points -= centroid
    points_angles = tf.math.atan2(proper_points[...,0], proper_points[...,1])
    points_angles -= points_angles[...,:1]
    points_angles = tf.where(points_angles < 0, points_angles + 2 * np.pi, points_angles)

    # sorting points by angle against origin point
    points_sort_idxs = tf.argsort(points_angles, axis=-1)
    sorted_points = tf.gather(proper_points, points_sort_idxs, axis=-2, batch_dims=batch_dims)

    origin_point = sorted_points[...,:1,:]
    p2_vertices = sorted_points[...,1:-1,:]
    p3_vertices = sorted_points[...,2:,:]

    triangle_areas = triangle_area(origin_point, p2_vertices, p3_vertices)

    return tf.reduce_sum(triangle_areas, axis=-2)

def calc_intersection_area(rect1, rect2, batch_dims=1, epsilon=1e-2):
    """Intersection area of two oriented rectangles.

    Pipeline:
      1) Compute all 16 line intersections between edges (infinite lines).
      2) Keep only points inside both rectangles; also add vertices that lie
         inside the other rectangle.
      3) Remove duplicates within a tolerance.
      4) Select up to 8 points (top-k mask) and form the convex polygon by
         sorting by angle around the centroid, then sum triangle fan areas.

    Args:
        rect1: Tensor (..., 4, 2). First rectangle.
        rect2: Tensor (..., 4, 2). Second rectangle.
        batch_dims: Batch dims passed to tf.gather for angle-sorting.
        epsilon: Tolerance for point-in-rect and duplicate filtering.

    Returns:
        Tensor (...) intersection areas.
    """
    rect1 = tf.convert_to_tensor(rect1, dtype=tf.float32)
    rect2 = tf.convert_to_tensor(rect2, dtype=tf.float32)
    
    # Find all intersection points
    points = parallel_intersection_points(rect1, rect2)
    
    # Filter points inside both rectangles
    mask1 = is_point_in_rect(points, rect1, epsilon)
    mask2 = is_point_in_rect(points, rect2, epsilon)
    mask = mask1 & mask2

    # Find vertices inside the other rectangle
    in_mask1 = is_point_in_rect(rect1, rect2, epsilon)
    in_mask2 = is_point_in_rect(rect2, rect1, epsilon)

    # Add vertices to the intersection points
    points = tf.concat([points, rect1, rect2], axis=-2)
    mask = tf.concat([mask, in_mask1, in_mask2], axis=-1)

    # Filter out duplicate points
    duplicate_mask = points_duplicates_mask(points)
    mask = tf.cast(mask, tf.int8) * duplicate_mask
    
    # Calculate the area of the polygon
    k_mask, k_points_idxs = tf.math.top_k(mask, k=8)
    k_points = tf.gather(points, k_points_idxs, axis=-2, batch_dims=batch_dims)
    k_points = tf.where(tf.math.is_nan(k_points) | tf.math.is_inf(k_points), tf.zeros_like(k_points), k_points)

    intersection_areas = convex_polygon_area(k_points, k_mask, batch_dims=batch_dims)
    
    return intersection_areas

def get_bbox_properties(bbox, angle):
    """Compute center and axis-aligned size of an oriented rectangle.

    Inverts rotation by angle to axis-align, then measures size as max-min per
    axis in the local frame.

    Args:
        bbox: Tensor (..., 4, 2). Rectangle corners.
        angle: Tensor broadcastable to (..., 1) in radians.

    Returns:
        Tuple (center, size, angle) where center is (..., 1, 2) and size is (..., 2).
    """
    bbox_center = tf.reduce_mean(bbox, axis=-2, keepdims=True)
    rot_matrix = gen_rot_matrix_yx(-angle)
    bbox_inv_rot = tf.matmul(bbox - bbox_center, rot_matrix)
    bbox_size = tf.reduce_max(bbox_inv_rot, axis=-2) - tf.reduce_min(bbox_inv_rot, axis=-2)
    return bbox_center, bbox_size, angle

def bbox_gaussian_properties(bbox, angle):
    """Map an oriented rectangle to a Gaussian mean and covariance.

    The mean is the box center u, and covariance is R * diag(size/2) * R^T, which
    produces an ellipse approximating the box footprint.

    Args:
        bbox: Tensor (..., 4, 2).
        angle: Tensor broadcastable to (..., 1) in radians.

    Returns:
        Tuple (u, sigma) with u (..., 1, 2) and sigma (..., 2, 2).
    """
    bbox_center, bbox_size, angle = get_bbox_properties(bbox, angle)
    u = bbox_center
    R = gen_rot_matrix_yx(angle)

    delta = tf.linalg.diag(bbox_size/2)

    sigma = tf.matmul(tf.matmul(R, delta), R, transpose_b=True)

    return u, sigma

# Function to invert a batch of 2x2 matrices
def values_of_2x2(matrices):
    """Extract elements (a,b,c,d) from 2x2 matrices.

    Args:
        matrices: Tensor (..., 2, 2).

    Returns:
        Tuple (a,b,c,d) each shape (...,).
    """
    a, b, c, d = matrices[..., 0, 0], matrices[..., 0, 1], matrices[..., 1, 0], matrices[..., 1, 1]
    return a, b, c, d

def det_2x2(matrices):
    """Determinant of 2x2 matrices.

    Args:
        matrices: Tensor (..., 2, 2).

    Returns:
        Tensor (...) determinants.
    """
    a, b, c, d = values_of_2x2(matrices)
    return a * d - b * c

def invert_2x2_matrices(matrices, return_det=False):
    """Inverse of 2x2 matrices with optional determinant return.

    Uses the closed-form inverse [[a,b],[c,d]]^{-1} = 1/det * [[d,-b],[-c,a]].

    Args:
        matrices: Tensor (..., 2, 2).
        return_det: If True, also returns the determinant.

    Returns:
        inv_matrices if return_det is False, else (inv_matrices, det).
    """
    a, b, c, d = values_of_2x2(matrices)
    det = a * d - b * c
    inv_det = tf.math.divide_no_nan(1.0, det)
    inv_matrices = tf.stack([d, -b, -c, a], axis=-1)
    inv_matrices = tf.reshape(inv_matrices, matrices.shape) * inv_det[..., tf.newaxis, tf.newaxis]
    if return_det:
        return inv_matrices, det
    return inv_matrices

class All2AllGrid(tf.keras.layers.Layer):
    """Build pairwise grids along a given axis by tiling and transposing.

    Given input x of shape Bx...xN_x..., returns (x1, x2) such that along the
    chosen axis every element is paired with every element, enabling all-to-all
    comparisons.
    """
    def __init__(self, axis, **kwargs):
        """Initialize layer.

        Args:
            axis: Integer axis along which to build the pairwise grid.
            **kwargs: Keras Layer kwargs.
        """
        super().__init__(**kwargs)

        self.axis = axis

    def build(self, input_shape):
        """Compute internal shapes and permutation for the grid transpose.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        self.input_shape = input_shape
        batch_dims = len(input_shape[:self.axis])
        head_dims = len(input_shape[self.axis:][1:])
        self.grid_dim = input_shape[self.axis]

        self.perm = list(range(batch_dims)) + [batch_dims+1, batch_dims] + [d+batch_dims+2 for d in list(range(head_dims))]

    def call(self, x):
        """Return two tensors that enumerate all pairwise combinations.

        Args:
            x: Tensor with size N along the configured axis.

        Returns:
            Tuple (x1, x2) where both have a new grid dimension replacing axis
            and list all pairwise pairs.
        """
        
        x1 = tf.repeat(tf.expand_dims(x, axis=self.axis), self.grid_dim, axis=self.axis)
        x2 = tf.transpose(x1, perm=self.perm)

        return x1, x2
    

def oriented_rectangle_area(x):
    """Twice the oriented area of a rectangle from three adjacent vertices.

    Uses polygon area formula on (v0, v1, v2). The absolute value matches the
    rectangle area.

    Args:
        x: Tensor (..., 3, 2) containing three consecutive vertices (y, x).

    Returns:
        Tensor (...) area magnitude.
    """
    y1, x1 = tf.split(x[...,0,:], 2, axis=-1)
    y2, x2 = tf.split(x[...,1,:], 2, axis=-1)
    y3, x3 = tf.split(x[...,2,:], 2, axis=-1)
    
    area = tf.abs(y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2))
    return area

class OrientedBboxIOU(tf.keras.layers.Layer):
    """Computes IoU between oriented rectangles via polygonal intersection.

    Intersection is computed by collecting edge intersections and contained
    vertices, deduplicating, sorting by angle and summing triangle areas.
    """
    def __init__(self, **kwargs):
        """Initialize layer.

        Args:
            **kwargs: Keras Layer kwargs.
        """
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Save the number of batch dims for later angle-based gather.

        Args:
            input_shape: Tuple (bbox1_shape, bbox2_shape).
        """
        self.batch_dims = len(input_shape[0][:-2])

    def call(self, x):
        """Compute IoU between two sets of oriented boxes.

        Args:
            x: Tuple (bbox1, bbox2) each (..., 4, 2).

        Returns:
            Tensor (...) IoU values.
        """
        # Unpack the input
        bbox1, bbox2 = x

        # Calculate the intersection area
        intersection_area = calc_intersection_area(bbox1, bbox2, batch_dims=self.batch_dims)

        # Calculate the union area
        area1 = oriented_rectangle_area(bbox1)
        area2 = oriented_rectangle_area(bbox2)
        union_area = area1 + area2 - intersection_area

        # Calculate the IoU
        iou = tf.math.divide_no_nan(intersection_area, union_area)

        return iou

def gaussian_iou(u1, sigma1, u2, sigma2):
    """Gaussian IoU proxy via Bhattacharyya coefficient.

    Approximates overlap of two Gaussian ellipses N(u1, S1) and N(u2, S2) by the
    Bhattacharyya coefficient: BC = exp(-Bd), with Bd the Bhattacharyya distance
    using S = (S1 + S2)/2. Converts to a probability-like IoU by Hellinger metric.

    Args:
        u1: Tensor (..., 1, 2). Mean of first Gaussian.
        sigma1: Tensor (..., 2, 2). Covariance of first Gaussian.
        u2: Tensor (..., 1, 2). Mean of second Gaussian.
        sigma2: Tensor (..., 2, 2). Covariance of second Gaussian.

    Returns:
        Tensor (...) in [0, 1] as a soft IoU measure.
    """
    sigma = (sigma1 + sigma2)/2
    u_diff = u1 - u2
    inv_sigma, det_sigma = invert_2x2_matrices(sigma, return_det=True)
    Bd = tf.einsum('...id,...dd,...id->...', u_diff/8, inv_sigma, u_diff)
    Bd += 1/2*tf.math.log(det_sigma/(det_2x2(sigma1)*det_2x2(sigma2))**0.5)
    Bc = tf.math.exp(-Bd)

    Hd = (1-Bc)**0.5

    prob_iou = 1-Hd

    return prob_iou

class GaussianOrientedBboxIou(tf.keras.layers.Layer):
    """Pairwise Gaussian IoU between oriented boxes mapped to Gaussians.

    Converts each oriented box to (u, Sigma), then evaluates the Gaussian IoU
    proxy for all-to-all pairs along the box axis.
    """
    def __init__(self, **kwargs):
        """Initialize layer.

        Args:
            **kwargs: Keras Layer kwargs.
        """
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Prepare grid layers for all-to-all comparisons.

        Args:
            input_shape: Tuple like (bboxes_shape, angles_shape).
        """
        self.u_grid_layer = All2AllGrid(axis=-3)
        self.sigma_grid_layer = All2AllGrid(axis=-3)

    def call(self, bboxes, angles):
        """Compute Gaussian IoU for all pairs of oriented boxes.

        Args:
            bboxes: Tensor (..., N, 4, 2). Oriented rectangles.
            angles: Tensor (..., N) angles in radians.

        Returns:
            Tensor (..., N, N) with Gaussian IoU proxy.
        """

        u, sigma = bbox_gaussian_properties(bboxes, angles)

        u1, u2 = self.u_grid_layer(u)
        sigma1, sigma2 = self.sigma_grid_layer(sigma)

        prob_iou = gaussian_iou(u1, sigma1, u2, sigma2)

        return prob_iou
    

def oriented_bbox_from_vec(vec, thickness):
    """Construct an oriented rectangle from a 2-point vector and thickness.

    Shifts endpoints by the normal direction scaled by thickness/2 to form a
    rectangle aligned with the vector direction.

    Args:
        vec: Tensor (..., 2, 2) with endpoints.
        thickness: Tensor broadcastable to (..., 1).

    Returns:
        Tuple (bbox, angle) where bbox is (..., 4, 2) and angle is (..., 1).
    """
    vec_diff = vec[...,0,:] - vec[...,1,:]
    vec_angle = tf.math.atan2(*tf.split(vec_diff, 2, axis=-1))
    perp_shift = tf.expand_dims(tf.concat([tf.math.cos(vec_angle), -tf.math.sin(vec_angle)], axis=-1) * thickness/2, axis=-2)
    p0, p1 = tf.split(vec + perp_shift, 2, axis=-2)
    p3, p2 = tf.split(vec - perp_shift, 2, axis=-2)

    return tf.concat([p0, p1, p2, p3], axis=-2), vec_angle

def oriented_square_bbox_from_vec(vec, zero_angle=False):
    """Square box derived from a vector, optionally axis-aligned.

    If zero_angle is True, returns an axis-aligned square centered at the vector
    midpoint whose side equals vector_length/sqrt(2). Otherwise, creates a square
    with diagonal aligned to the vector.

    Args:
        vec: Tensor (..., 2, 2) endpoints.
        zero_angle: Bool flag for axis-aligned square.

    Returns:
        Tuple (bbox, angle) where bbox is (..., 4, 2) and angle is (..., 1).
    """
    vec_diff = vec[...,0,:] - vec[...,1,:] # [...,2]

    vec_len = tf.norm(vec_diff, axis=-1, keepdims=True)/2**0.5 # [...,1]
    if zero_angle:
        vec_angle = tf.zeros_like(vec_len)
        vec_center = tf.reduce_mean(vec, axis=-2, keepdims=True) #[...,2]
        shift = tf.constant([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]], dtype=vec_center.dtype)*vec_len[...,tf.newaxis]
        bbox = vec_center + shift
    else:
        vec_angle = tf.math.atan2(*tf.split(vec_diff, 2, axis=-1))
        shift_angle = vec_angle + np.pi/4
        perp_shift = tf.expand_dims(tf.concat([tf.math.cos(shift_angle), -tf.math.sin(shift_angle)], axis=-1) * vec_len, axis=-2)
        p0, p2 = tf.split(vec, 2, axis=-2)
        p1 = p0 + perp_shift
        p3 = p2 - perp_shift
        bbox = tf.concat([p0, p1, p2, p3], axis=-2)

    return bbox, vec_angle

def line_vec_gaussian_inputs(vec, thickness):
    """Gaussian parameters for a thick line segment.

    Encodes the segment as a Gaussian where size is [thickness, length] and
    angle is aligned with the segment direction.

    Args:
        vec: Tensor (..., 2, 2) endpoints.
        thickness: Tensor (..., 1) thickness.

    Returns:
        Tuple (center, size, angle) with shapes (..., 1, 2), (..., 2), (...,).
    """
    vec_diff = vec[...,0,:] - vec[...,1,:]

    vec_angle = tf.squeeze(tf.math.atan2(*tf.split(vec_diff, 2, axis=-1)), axis=-1)

    vec_len = tf.norm(vec_diff, axis=-1, keepdims=True)
    vec_size = tf.concat([thickness, vec_len], axis=-1)

    vec_center = tf.reduce_mean(vec, axis=-2, keepdims=True)

    return vec_center, vec_size, vec_angle

def straight_bbox_vec_gaussian_inputs(vec):
    """Axis-aligned square Gaussian approx for a segment.

    Side length equals vector_length/sqrt(2) and angle is zero.

    Args:
        vec: Tensor (..., 2, 2).

    Returns:
        Tuple (center, size, angle) with shapes (..., 1, 2), (..., 2), (...,).
    """
    vec_center = tf.reduce_mean(vec, axis=-2, keepdims=True)
    vec_size = tf.repeat(tf.norm(vec[...,0,:] - vec[...,1,:], axis=-1, keepdims=True)/2**0.5, 2, axis=-1)
    vec_angle = tf.zeros(vec_size.shape[:-1], dtype=vec_size.dtype)

    return vec_center, vec_size, vec_angle

def vec_bbox_gaussian_properties(bbox_center, bbox_size, angle):
    """Build Gaussian mean and covariance from center, size and angle.

    Covariance is constructed as R * diag(size/2) * R^T.

    Args:
        bbox_center: Tensor (..., 1, 2).
        bbox_size: Tensor (..., 2).
        angle: Tensor (...,) or broadcastable.

    Returns:
        Tuple (u, sigma) with u (..., 1, 2) and sigma (..., 2, 2).
    """
    u = bbox_center
    R = gen_rot_matrix_yx(angle)

    delta = tf.linalg.diag(bbox_size/2)

    sigma = tf.matmul(tf.matmul(R, delta), R, transpose_b=True)

    return u, sigma

def random_vecs_generator(batch_shape, vec_range, thickness_range):
    """Generate random 2-point vectors and thickness values.

    Args:
        batch_shape: Tuple of batch dims for output tensors.
        vec_range: Float stddev for normal sampling of endpoints.
        thickness_range: Tuple (min, max) for uniform thickness sampling.

    Returns:
        Tuple (vecs, thickness) where vecs is (..., 2, 2) and thickness is (..., 1).
    """
    vecs = tf.random.normal(batch_shape + (2, 2), mean=0, stddev=vec_range)
    thickness = tf.random.uniform(batch_shape + (1,), minval=thickness_range[0], maxval=thickness_range[1])
    return vecs, thickness

class GaussianVecIou(tf.keras.layers.Layer):
    """Pairwise Gaussian IoU for line segments encoded as Gaussians.

    Segments can be modeled as thick lines ([thickness, length]) or axis-aligned
    squares (length/sqrt(2), length/sqrt(2)) depending on lines_mask.
    """
    def __init__(self, **kwargs):
        """Initialize layer.

        Args:
            **kwargs: Keras Layer kwargs.
        """
        super().__init__(**kwargs)

    def build(self, vecs_shape):
        """Prepare all-to-all grid layers for means and covariances.

        Args:
            vecs_shape: Shape of input vectors.
        """
        self.u_grid_layer = All2AllGrid(axis=-3)
        self.sigma_grid_layer = All2AllGrid(axis=-3)

    def straight_bbox_vec_gaussian_inputs(self, vec):
        """Convenience: axis-aligned square Gaussian inputs for a segment.

        Args:
            vec: Tensor (..., 2, 2).

        Returns:
            Tuple (center, size, angle) with zero angle.
        """
        vec_center = tf.reduce_mean(vec, axis=-2, keepdims=True)
        vec_size = tf.repeat(tf.norm(vec[...,0,:] - vec[...,1,:], axis=-1, keepdims=True)/2**0.5, 2, axis=-1)

        return vec_center, vec_size, self.zero_angle 
    
    @staticmethod
    def calc_bbox_center(vec):
        """Center point of a 2-point vector.

        Args:
            vec: Tensor (..., 2, 2).

        Returns:
            Tensor (..., 1, 2) center.
        """
        return tf.reduce_mean(vec, axis=-2, keepdims=True)
    
    @staticmethod
    def calc_bbox_size_and_angle(vec, thickness, lines_mask):
        """Compute Gaussian size and angle for segments with optional thickness.

        When lines_mask=1, uses [thickness, length] and the segment angle.
        When lines_mask=0, uses a square of side length/sqrt(2) and zero angle.

        Args:
            vec: Tensor (..., 2, 2).
            thickness: Tensor (..., 1).
            lines_mask: Tensor (...,) with 1 for line model, 0 for square model.

        Returns:
            Tuple (bbox_size, angle) with shapes (..., 2) and (...,).
        """
        vec_diff = vec[...,0,:] - vec[...,1,:]

        # bbox size
        vec_len = tf.norm(vec_diff, axis=-1, keepdims=True)
        line_vec_size = tf.concat([thickness, vec_len], axis=-1)

        bbox_vec_size = tf.repeat(tf.norm(vec_diff, axis=-1, keepdims=True)/2**0.5, 2, axis=-1)

        bbox_size = line_vec_size * lines_mask + bbox_vec_size * (1-lines_mask)

        # bbox angle
        vec_angle = tf.squeeze(tf.math.atan2(*tf.split(vec_diff, 2, axis=-1))*lines_mask, axis=-1)

        return bbox_size, vec_angle

    def call(self, vecs, thickness, lines_mask):
        """Compute pairwise Gaussian IoU between vectors.

        Args:
            vecs: Tensor (..., N, 2, 2) endpoints per vector.
            thickness: Tensor (..., N, 1) thickness per vector.
            lines_mask: Tensor (..., N) selecting line vs square model.

        Returns:
            Tensor (..., N, N) Gaussian IoU proxy for all pairs.
        """
        bbox_center = self.calc_bbox_center(vecs)
        bbox_size, angles = self.calc_bbox_size_and_angle(vecs, thickness, lines_mask)

        u, sigma = vec_bbox_gaussian_properties(bbox_center, bbox_size, angles)

        u1, u2 = self.u_grid_layer(u)
        sigma1, sigma2 = self.sigma_grid_layer(sigma)

        prob_iou = gaussian_iou(u1, sigma1, u2, sigma2)

        return prob_iou
    

class GatherBestExamples(tf.keras.layers.Layer):
    """Gather attributes at the position of the max score along a given axis.

    Returns squeezed tensors per attribute, and optionally the argmax indices.
    """
    def __init__(self, axis, return_idxs=False, **kwargs):
        """Initialize layer.

        Args:
            axis: Integer axis along which to take argmax.
            return_idxs: If True, also return indices.
            **kwargs: Keras Layer kwargs.
        """
        super().__init__(**kwargs)

        self.axis = axis
        self.return_idxs = return_idxs

    def build(self, scores_shape, attributes_shape):
        """Prepare internal helpers based on input shapes.

        Args:
            scores_shape: Shape of the scores tensor.
            attributes_shape: List of shapes for attribute tensors.
        """
        self.abs_axis = self.axis if self.axis >= 0 else len(scores_shape) + self.axis
        
        self.squeeze_layers = [SqueezeImg(name=f'{self.name}-SqImg_i') for i in range(len(attributes_shape))]

    def call(self, scores, attributes):
        """Select and gather attributes at max-score positions.

        Args:
            scores: Tensor with scores along axis.
            attributes: List of tensors aligned with scores along axis.

        Returns:
            List of squeezed attribute tensors, optionally followed by indices.
        """
        max_idx = tf.argmax(scores, axis=self.abs_axis)

        outputs =  [sq_l(tf.squeeze(tf.gather(a, max_idx, axis=self.abs_axis, batch_dims=self.abs_axis), axis=self.abs_axis)) for sq_l, a in zip(self.squeeze_layers, attributes)]

        if self.return_idxs:
            outputs.append(max_idx)

        return outputs
    
class UnflattenVec(tf.keras.layers.Layer):
    """Reshape a flat last-dimension vector into (..., N, 2)."""
    def build(self, input_shape):
        """Compute target shape from the input's last dimension.

        Args:
            input_shape: Shape where last dim equals 2*N.
        """
        batch_dims = input_shape[:-1]
        points_num = input_shape[-1]//2

        self.target_shape = batch_dims + (points_num, 2)

    def call(self, x):
        """Reshape x to (..., N, 2).

        Args:
            x: Tensor with shape computed in build.

        Returns:
            Reshaped tensor with last two dims (N, 2).
        """
        return tf.reshape(x, self.target_shape)
    
class BinaryClassMasks(tf.keras.layers.Layer):
    """Convert logits to one-hot masks per class split along the last axis."""

    def build(self, input_shape):
        """Record number of classes and constant splits for tf.split.

        Args:
            input_shape: Shape where last dim equals number of classes.
        """
        self.num_classes = input_shape[-1]
        self.splits = tf.ones((self.num_classes,), dtype=tf.int32)

    def call(self, x):
        """Return a list of binary masks, one per class.

        Args:
            x: Tensor (..., C) logits or probabilities.

        Returns:
            List of C tensors, each (..., 1), from splitting one-hot over classes.
        """
        return tf.split(tf.one_hot(tf.argmax(x, axis=-1), self.num_classes), self.splits, axis=-1)
    
class MaskApply(tf.keras.layers.Layer):
    """Apply a binary mask (or its inverse) to an input tensor by multiplication."""
    def __init__(self, inv=False, **kwargs):
        """Initialize layer.

        Args:
            inv: If True, apply 1-mask instead of mask.
            **kwargs: Keras Layer kwargs.
        """
        super().__init__(**kwargs)
        self.inv = inv  

    def call(self, x, mask):
        """Multiply input by mask (or inverse mask).

        Args:
            x: Tensor to be masked.
            mask: Broadcastable binary mask.

        Returns:
            Tensor with the same shape as x after masking.
        """
        mask = tf.cast(mask, x.dtype)
        if self.inv:
            mask = 1. - mask
        return x * mask
    
class OverlapsNMS(tf.keras.layers.Layer):
    """Batched NMS using precomputed overlaps (IoU) and scores.

    Wraps tf.image.non_max_suppression_overlaps for each batch element, then
    gathers and pads selected attributes to a fixed max_output_size.
    """
    def __init__(self, max_output_size, overlap_threshold, score_threshold, return_indices, parallel_iterations=8, **kwargs):
        """Initialize layer.

        Args:
            max_output_size: Max boxes to keep per batch element.
            overlap_threshold: IoU threshold for suppression.
            score_threshold: Minimum score to consider.
            return_indices: If True, also return selected indices.
            parallel_iterations: map_fn parallelism.
            **kwargs: Keras Layer kwargs.
        """
        super().__init__(**kwargs)

        self.max_output_size = max_output_size
        self.overlap_threshold = overlap_threshold
        self.score_threshold = score_threshold
        self.return_indices = return_indices
        self.parallel_iterations = parallel_iterations

    def build(self, overlaps_shape, scores_shape, attributes_shape):
        """Prepare output dtypes and pad configurations.

        Args:
            overlaps_shape: Shape of overlaps (..., N, N).
            scores_shape: Shape of scores (..., N).
            attributes_shape: List of attribute shapes (..., N, ...).
        """
        self.attributes_num = len(attributes_shape)

        self.map_output_sign = [tf.float32]*(self.attributes_num+1)

        self.paddings_list = [tf.zeros((len(s)-2, 2), dtype=tf.int32) for s in attributes_shape] + [tf.zeros((0,2), dtype=tf.int32)]
        self.front_pad = tf.constant([[0, 1]], dtype=tf.int32)

        if self.return_indices:
            self.map_output_sign += [tf.int32]
            self.paddings_list += [tf.zeros((0,2), dtype=tf.int32)]


    def _get_paddings(self, pad_size):
        """Construct paddings to extend outputs to max_output_size.

        Args:
            pad_size: Integer number of rows to add at front.

        Returns:
            List of paddings tensors for each output.
        """
        front_pad = self.front_pad*pad_size

        return [tf.concat([front_pad, pl], axis=0) for pl in self.paddings_list]
    
    def _gather_selected(self, tensor, indices):
        """Gather rows at selected indices along axis=0 with batch_dims=0.

        Args:
            tensor: Tensor with first dim N.
            indices: 1D tensor of selected indices.

        Returns:
            Tensor of gathered rows.
        """
        return tf.gather(tensor, indices, axis=0, batch_dims=0)

    def _non_max_suppresion(self, overlaps, scores, attributes):
        """Run NMS for a single batch element and pad outputs.

        Args:
            overlaps: Tensor (N, N) overlaps/IoU.
            scores: Tensor (N,) scores.
            attributes: List of attribute tensors with first dim N.

        Returns:
            List of selected attributes, a mask (1 for kept rows), and optionally indices.
        """
        nms_indices = tf.image.non_max_suppression_overlaps(overlaps, scores, self.max_output_size, self.overlap_threshold, self.score_threshold)

        pad_size = self.max_output_size - tf.shape(nms_indices)[0]
        paddings = self._get_paddings(pad_size)

        mask = tf.ones_like(nms_indices, dtype=tf.float32)

        selected_attributes = [self._gather_selected(attr, nms_indices) for attr in attributes]
        selected_attributes.append(mask)

        if self.return_indices:
            selected_attributes.append(nms_indices)

        selected_attributes = [tf.pad(sa, p) for sa, p in zip(selected_attributes, paddings)]

        return selected_attributes

    def call(self, overlaps, scores, attributes):
        """Apply NMS to each batch element in parallel using map_fn.

        Args:
            overlaps: Tensor (B, N, N) overlaps.
            scores: Tensor (B, N) scores.
            attributes: List of tensors (B, N, ...).

        Returns:
            Padded selected attributes per batch and a selection mask; optionally indices.
        """
        return tf.map_fn(lambda x: self._non_max_suppresion(*x), (overlaps, scores, attributes), dtype=self.map_output_sign, parallel_iterations=self.parallel_iterations)