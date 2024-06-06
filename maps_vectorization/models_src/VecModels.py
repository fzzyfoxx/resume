import tensorflow as tf
import math
import numpy as np
from models_src.Hough import VecDrawer
from models_src.fft_lib import xy_coords
from models_src.Attn_variations import UnSqueezeImg, SqueezeImg


def tensor_from_coords(coords, values, size=32):
    x = tf.SparseTensor(coords, values, (size,size))
    x = tf.sparse.reorder(x)
    x = tf.sparse.to_dense(x, validate_indices=False)

    return x

def one_hot_angles(x, splits):
    bins = tf.cast(tf.round(x/(2*math.pi)*splits) + splits//2, tf.int32)
    bins = tf.where(bins==splits, 0, bins)
    return tf.one_hot(bins, splits)

def prepare_endpoint_label(vecs_col, vecs_mask, angle_splits=8, size=32):
    idxs = tf.where(vecs_mask>0)
    if len(idxs)>0:
        v = tf.gather_nd(vecs_col, idxs)
        v0, v1 = [a[:,0] for a in tf.split(v[...,::-1], 2, axis=-2)]

        v_diff = (v1-v0)
        angles0 = tf.math.atan2(*tf.split(tf.cast(v_diff, tf.float32), 2, axis=-1))
        angles1 = tf.math.atan2(*tf.split(tf.cast(-v_diff, tf.float32), 2, axis=-1))

        mask_values = tf.ones((len(v)*2,))

        v = tf.concat([v0,v1], axis=0)
        angles = tf.concat([angles0, angles1], axis=0)[:,0]

        mask_label = tf.expand_dims(tensor_from_coords(v, mask_values, size=size), axis=-1)
        angle_label = tensor_from_coords(v, angles, size=size)
        angle_label = one_hot_angles(angle_label, splits=angle_splits)

        label = tf.concat([mask_label, mask_label, angle_label*mask_label], axis=-1)
    else:
        label = tf.zeros((size, size, 2+angle_splits))
    return label

def dist_to_line(points, vecs):
    v1, v2 = tf.split(vecs, 2, axis=-2)
    y1, x1 = tf.split(v1-1e-4, 2, axis=-1)
    y2, x2 = tf.split(v2+1e-4, 2, axis=-1)
    y0, x0 = tf.split(points, 2, axis=-1)
    
    dist = tf.abs((x2-x1)*(y0-y1) - (x0-x1)*(y2-y1))/(((x2-x1)**2 + (y2-y1)**2)**0.5 + 1e-4)
    
    return dist

def gen_visible_endpoints(pattern_mask, pattern_vecs_mask, pattern_vecs_col, n, masks, filter=True):
    cov_mask = pattern_mask*(1-tf.reduce_max(masks[(n+1):], axis=0))
    idxs = tf.where(pattern_vecs_mask>0)
    v = tf.gather_nd(pattern_vecs_col, idxs)[...,::-1]

    if filter:
        point_line_filter = tf.math.reduce_all(v[:,0]==v[:,1], axis=-1)

        v_filtered = tf.cast(tf.gather(v, tf.where(point_line_filter==True)[:,0], axis=0), tf.int64)
        v = tf.gather(v, tf.where(point_line_filter==False)[:,0], axis=0)
        if len(v_filtered)>0:
            temp_filter = 1-tf.cast(tensor_from_coords(tf.reshape(v_filtered, (-1,2)), tf.ones((len(v_filtered)*2,)))[...,tf.newaxis], tf.float32)
            cov_mask *= temp_filter

    pattern_points = tf.pad(tf.cast(tf.where(cov_mask[...,0]>0), tf.float32), [[1,0],[0,0]], constant_values=-1)
    dists = dist_to_line(pattern_points[tf.newaxis], v)[...,0]
    lines_assignment = tf.argmin(dists, axis=0)
    assignment_mask = tf.pad(tf.transpose(tf.one_hot(lines_assignment, len(v)), perm=[1,0])[:,1:], [[0,0],[1,0]])

    points_pos = tf.argmax((tf.reduce_sum((v[:,:,tf.newaxis]-pattern_points[tf.newaxis, tf.newaxis])**2+1e-4, axis=-1)**0.5-dists[:,tf.newaxis]**2)*assignment_mask[:,tf.newaxis], axis=-1)
    shifted_endpoints = tf.gather(pattern_points, points_pos, axis=0)[:,::-1]

    new_endpoints = tf.reshape(shifted_endpoints, (-1,2))
    endpoints_idxs = tf.where(tf.reshape(points_pos, (-1,))>0)[:,0]
    new_endpoints = tf.reshape(tf.gather(new_endpoints, endpoints_idxs, axis=0), (-1,2,2))

    if filter:
        point_line_filter = tf.math.reduce_all(new_endpoints[:,0]==new_endpoints[:,1], axis=-1)
        new_endpoints = tf.reshape(tf.gather(new_endpoints, tf.where(point_line_filter==False)[:,0], axis=0), (-1,2))

    return new_endpoints

def gen_visible_endpoints_masks(pattern_mask, pattern_vecs_mask, pattern_vecs_col, n, masks, size=32):
    new_endpoints = gen_visible_endpoints(pattern_mask, pattern_vecs_mask, pattern_vecs_col, n, masks)
    if len(new_endpoints)>0:
        new_label = tensor_from_coords(tf.cast(new_endpoints, tf.int64), tf.ones((len(new_endpoints),)), size=size)[...,tf.newaxis]
    else:
        new_label = tf.zeros((size, size, 1))

    return new_label


def extract_lines_from_endpoints(endpoints, endpoint_contingency, threshold=0.5):
    pair_coords = tf.where(endpoint_contingency>threshold)
    p0 = tf.gather(endpoints, pair_coords[...,0], axis=0)
    p1 = tf.gather(endpoints, pair_coords[...,1], axis=0)
    lines = tf.stack([p0,p1], axis=0)
    return lines

def gen_vec_assignment_mask(pattern_vecs_mask, pattern_vecs_col, cov_mask):
    idxs = tf.where(pattern_vecs_mask>0)
    v = tf.gather_nd(pattern_vecs_col, idxs)[...,::-1]

    pattern_points = tf.cast(tf.where(cov_mask[...,0]>0), tf.float32)
    dists = dist_to_line(pattern_points[tf.newaxis], v)[...,0]
    lines_assignment = tf.argmin(dists, axis=0)
    assignment_mask = tf.transpose(tf.one_hot(lines_assignment, len(v)), perm=[1,0])

    return v, pattern_points, assignment_mask

def random_pattern_point(pattern_vecs_mask, pattern_vecs_col, cov_mask):
    v, pattern_points, assignment_mask = gen_vec_assignment_mask(pattern_vecs_mask, pattern_vecs_col, cov_mask)

    available_vecs_idxs = tf.where(tf.reduce_sum(assignment_mask, axis=-1)>0)[:,0]
    random_vec_idx = tf.py_function(np.random.randint, [0,len(available_vecs_idxs)], tf.int32)
    v_n = available_vecs_idxs[random_vec_idx]

    selected_v = v[v_n]
    selected_points = tf.gather(pattern_points, tf.where(assignment_mask[v_n]>0)[:,0], axis=0)
    selected_dists = dist_to_line(selected_points, selected_v)[...,0]

    points_pos = tf.argmax((tf.reduce_sum((selected_v[:,tf.newaxis]-selected_points[tf.newaxis])**2+1e-4, axis=-1)**0.5-selected_dists[tf.newaxis]**2), axis=-1)
    shifted_endpoints = tf.gather(selected_points, points_pos, axis=0)[::-1]

    random_pt_idx = tf.py_function(np.random.randint, [0,len(selected_points)], tf.int32)
    selected_point = selected_points[random_pt_idx]

    return shifted_endpoints, selected_point

def closest_point_on_line(line_vec, points):

    p1 = tf.gather(line_vec, 0, axis=-2)-1e-4
    p2 = tf.gather(line_vec, 1, axis=-2)-1e-4
    p0 = points
    vec_angle = tf.math.atan2(*tf.split(p2-p1, 2, axis=-1))
    line_direction = tf.concat([tf.sin(vec_angle), tf.cos(vec_angle)], axis=-1)
    vector_to_object = p0-p1
    distance = tf.reduce_sum(vector_to_object*line_direction, axis=-1, keepdims=True)

    closest_position = p1 + distance*line_direction

    return closest_position

def calc_loss_slope(label, loss, n, values_range, sample_weight=None):
    shape = label.shape
    B = shape[0]
    a = tf.linspace(label, tf.random.uniform(shape, *values_range), n)
    batch_samples = tf.unstack(a, B, axis=1)
    batch_labels = tf.split(label, B, axis=0)

    values = [loss(b_l, b_s, sample_weight=sample_weight[b,tf.newaxis] if sample_weight is not None else None) for b_l, b_s, b in zip(batch_labels, batch_samples, range(B))]

    mean_values = tf.reduce_mean(tf.stack(values, axis=0), axis=0)

    return values, mean_values


class PixelMaskMapLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1., name='PMM', reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

        self.squeeze_img = SqueezeImg()
        self.gamma = gamma

    def get_config(self):
        return {
            'name': self.name,
            'reduction': self.reduction,
            'gamma': self.gamma
        }
    
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(self.squeeze_img(y_true), axis=-1)
        y_true_norm = y_true/(tf.reduce_sum(y_true, axis=-1, keepdims=True)+1e-6)
        y_true = tf.matmul(y_true, y_true_norm, transpose_a=True)

        diffs = tf.abs(y_true-y_pred)**self.gamma

        score = tf.reduce_mean(diffs, axis=-1)

        return score





class VecEndpointsDataset():
    def __init__(self, 
                 flg, 
                 batch_size, 
                 max_examples_num=3, 
                 min_width=0.2, 
                 min_lines_num=2, 
                 parallel_calls=4, 
                 size=32, 
                 color_rand_range=0.2,
                 angle_splits=8,
                 endpoint_blur_phases=2,
                 output_type=0,
                 angle_rand_range=15):

        self.flg = flg
        self.batch_size = batch_size
        self.max_examples_num = max_examples_num
        self.vd = VecDrawer(min_width=min_width, min_num=min_lines_num)
        self.parallel_calls = parallel_calls
        self.size = size
        self.angle_splits = angle_splits

        self.color_range = color_rand_range

        self.endpoint_blur_phases = endpoint_blur_phases

        self.output_type=output_type

        self.angle_range = angle_rand_range/180*math.pi
        #0 img - endpoint mask, lines mask, angles splits masks
        #1 img, endpoints coords - endpoints contingency matrix
        #2 img, angle, sample_point - vec
        #3 img - lines mask, lines angle, lines thickness, lines center vec
        #4 img - padded visible masks

        if output_type==1:
            self.endpoints_xy_mask = self._gen_endpoints_xy_mask()

    def _gen_parameters(self, *args):
        n = tf.py_function(np.random.randint, [1,self.max_examples_num+1], [tf.int32])[0]
        Xuv, phase = self.flg.random_optimized_freqs(examples_num=n)
        vecs_col, lengths, lines_mask, vecs_slope, vecs_bias, Iuvx, vec_angle = self.flg.gen_vecs(Xuv, phase)

        return Xuv, vecs_col, lengths, lines_mask, vec_angle, phase, Iuvx,n
    
    def get_freq_features(self, Xuv):
        vecx, vecy = tf.split((self.size/Xuv), 2, axis=-1)

        vec_angle = tf.math.atan2(vecy,vecx)

        slope = tf.tan(-vec_angle)
        lengths = tf.abs(-slope*vecx)/(slope**2+1)**0.5
        return tf.concat([vec_angle, lengths], axis=-1)
    
    def _lines_one_hot_angles(self, masks, vec_angle):
        opp_vec_angle = vec_angle - tf.sign(vec_angle)*math.pi
        angles = tf.concat([vec_angle, opp_vec_angle], axis=-1)
        angles = tf.reduce_max(one_hot_angles(angles, self.angle_splits), axis=-2)[:,tf.newaxis, tf.newaxis]

        lines_label = tf.reduce_max(tf.concat([masks, angles*masks], axis=-1), axis=0)

        return lines_label
    
    def _endpoint_angle_pooling(self, endpoint_label, masks):
        angles = tf.reduce_sum(tf.reshape(endpoint_label[...,2:], (-1,self.size**2, self.angle_splits)), axis=1)[:,tf.newaxis, tf.newaxis]
        angles_left, angles_right = tf.split(angles, 2, axis=-1)
        angles_left = tf.argmax(angles_left, axis=-1)
        angles_right = tf.argmax(angles_right, axis=-1)+self.angle_splits//2
        angles = tf.reduce_max(tf.one_hot(tf.stack([angles_left, angles_right], axis=-1), depth=self.angle_splits), axis=-2)
        lines_label = tf.reduce_max(tf.concat([masks*0, masks, angles*masks], axis=-1), axis=0)

        return lines_label
    
    def _endpoint_label_blur(self, endpoint_label, masks):
        for i in range(self.endpoint_blur_phases):
            endpoint_label = tf.nn.max_pool2d(endpoint_label, 3, 1, padding='SAME')
            endpoint_label = tf.nn.avg_pool2d(endpoint_label, 3, 1, padding='SAME')
        endpoint_label *= masks

        return endpoint_label
    
    def _padded_endpoints(self, pattern_mask, pattern_vecs_mask, pattern_vecs_col, n, masks):
        endpoints = gen_visible_endpoints(pattern_mask, pattern_vecs_mask, pattern_vecs_col, n, masks)
        endpoints_num = tf.reduce_min(tf.stack([len(endpoints), self.size], axis=0))

        pad_size = self.size-endpoints_num

        endpoints_mask = tf.pad(tf.ones((endpoints_num,1)), [[0, pad_size],[0,0]])
        endpoints = tf.pad(endpoints[:endpoints_num], [[0, pad_size],[0,0]])

        return endpoints, endpoints_mask
    
    def _gen_endpoints_xy_mask(self):
        xy = xy_coords((self.size*self.max_examples_num,self.size*self.max_examples_num))
        p0_mask = tf.where(xy[...,0]%2==0, 1, 0)
        p1 = tf.where(xy[...,1]==xy[...,0]-1, 1, 0)*(1-p0_mask)
        p0 = tf.where(xy[...,1]==xy[...,0]+1, 1, 0)*p0_mask
        endpoints_xy_mask = tf.cast(p1+p0, tf.float32)

        return endpoints_xy_mask
    
    def _gen_endpoints_cross_label(self, endpoints_mask):
        return self.endpoints_xy_mask*tf.matmul(endpoints_mask, endpoints_mask, transpose_b=True)
    
    '''def _random_visible_mask(self, masks):
        n = tf.py_function(np.random.randint, [0,len(masks)-1], tf.int32)
        vis_mask = masks[n]-tf.reduce_max(masks[n+1:], axis=0)*masks[n]

        return vis_mask, n'''
    
    def _visible_mask(self, n, masks):
        vis_mask = masks[n]*(1-tf.reduce_max(masks[n+1:], axis=0))

        return vis_mask
    
    def _visible_masks(self, masks, min_points=1):
        vis_masks = tf.map_fn(lambda x: self._visible_mask(*x, masks), [tf.range(len(masks[:-1]))], fn_output_signature=tf.float32)
        vis_masks_flat = tf.reshape(vis_masks, (-1, self.size**2))

        points_sum = tf.reduce_sum(vis_masks_flat, axis=-1)
        idxs = tf.where(points_sum>=min_points)[:,0]

        vis_masks = tf.gather(vis_masks, idxs, axis=0)

        return vis_masks, idxs

    def _random_mask_point(self, mask):
        points = tf.where(mask>0.5)
        n = tf.py_function(np.random.randint, [0,len(points)], tf.int32)

        return tf.cast(points[n], tf.float32)

    def _closest_vec_selection(self, vecs, points):
        #filtered_vecs = tf.gather(vecs_col, tf.where(vecs_mask>0), axis=0)
        dists = tf.squeeze(dist_to_line(points, vecs))

        selected_vec = tf.gather(vecs, tf.argmin(dists), axis=0)
        return selected_vec
    
    def _random_visible_mask(self, masks, min_points=1):
        vis_masks = tf.map_fn(lambda x: self._visible_mask(*x, masks), [tf.range(len(masks[:-1]))], fn_output_signature=tf.float32)
        vis_masks = tf.reshape(vis_masks, (-1, self.size**2))

        points_sum = tf.reduce_sum(vis_masks, axis=-1)
        idxs = tf.where(points_sum>=min_points)[:,0]

        n = tf.py_function(np.random.randint, [0,len(idxs)], tf.int32)
        n = idxs[n]
        return tf.reshape(vis_masks[n], (self.size, self.size, 1)), n
    
    def _gen_center_vecs_map(self, vec_mask, vec_col, vis_mask):
        v, pattern_points, assignment_mask = gen_vec_assignment_mask(vec_mask, vec_col, cov_mask=vis_mask)
        closest_position = closest_point_on_line(v[tf.newaxis], pattern_points[:,tf.newaxis])
        closest_position = tf.gather(closest_position, tf.argmax(assignment_mask, axis=0), axis=1, batch_dims=1)
        center_vec = closest_position-pattern_points

        int_pts = tf.cast(pattern_points, tf.int64)
        center_vec_map = tf.stack([tensor_from_coords(int_pts, center_vec[:,0], size=32),
                                tensor_from_coords(int_pts, center_vec[:,1], size=32)], axis=-1)
        
        return center_vec_map
    
    def _vec_length_filter(self, vecs_col, min_length=1):
        return tf.where(tf.reduce_sum((vecs_col[...,0,:]-vecs_col[...,1,:])**2, axis=-1)**0.5>=min_length, 1., 0.)

    
    def _map_drawing(self, Xuv, vecs_col, lengths, lines_mask, vec_angle, phase, Iuvx,n):
        I, masks, vecs_mask, vecs_col, colors, thickness = tf.py_function(self.vd.draw_vecs, [vecs_col, lines_mask, lengths], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        I.set_shape((self.size,self.size,3))

        I = tf.clip_by_value(I+tf.random.uniform((self.size, self.size, 1), -self.color_range, self.color_range), 0., 1.)

        masks = tf.concat([masks[:-1], tf.zeros((1,self.size, self.size, 1))], axis=0)
        if self.output_type==0:

            endpoint_label = tf.map_fn(lambda x: gen_visible_endpoints_masks(*x, masks, size=32), [masks[:-1], vecs_mask[:-1], vecs_col[:-1], tf.range(len(vecs_col[:-1]))], fn_output_signature=tf.float32)
            endpoint_label = self._endpoint_label_blur(endpoint_label, masks[:-1])
            endpoint_label = tf.reduce_max(endpoint_label, axis=0)
            line_label = self._lines_one_hot_angles(masks[:-1], vec_angle)
            line_label, angle_label = tf.split(line_label, [1,self.angle_splits], axis=-1)
            #label = tf.concat([endpoint_label, line_label], axis=-1)

            return I, endpoint_label, line_label, angle_label
        
        elif self.output_type==1:

            endpoints, endpoints_mask = tf.map_fn(lambda x: self._padded_endpoints(*x, masks), [masks[:-1], vecs_mask[:-1], vecs_col[:-1], tf.range(len(vecs_col[:-1]))], fn_output_signature=(tf.float32, tf.float32))
            endpoints = tf.reshape(endpoints, (-1,2))
            endpoints_mask = tf.reshape(endpoints_mask, (-1,1))

            pad_size = self.size*self.max_examples_num - len(endpoints)
            endpoints = tf.pad(endpoints, [[0,pad_size],[0,0]])
            endpoints_mask = tf.pad(endpoints_mask, [[0, pad_size],[0,0]])
            endpoints_cross_label = self._gen_endpoints_cross_label(endpoints_mask)
            return I, endpoints, endpoints_mask, endpoints_cross_label
        
        elif self.output_type==2:
            
            vis_mask, n = self._random_visible_mask(masks, min_points=2)

            '''point = self._random_mask_point(vis_mask[...,0])
            visible_endpoints = gen_visible_endpoints(masks[n], vecs_mask[n], vecs_col[n], n, masks, filter=False)
            selected_vec = self._closest_vec_selection(visible_endpoints, point[tf.newaxis])'''

            selected_vec, point = random_pattern_point(vecs_mask[n], vecs_col[n], vis_mask)
            
            angle = vec_angle[n] + tf.random.uniform( (1,),-self.angle_range, self.angle_range)

            return I, -angle, point, selected_vec
        
        elif self.output_type==3:
            
            vis_masks, vis_idxs = self._visible_masks(masks, min_points=2)
            vecs_col = tf.gather(vecs_col, vis_idxs, axis=0)
            vecs_mask = self._vec_length_filter(vecs_col, min_length=1)
            vecs_col *= vecs_mask[...,tf.newaxis, tf.newaxis]
            vec_angle = tf.gather(vec_angle, vis_idxs, axis=0)
            thickness = tf.gather(thickness, vis_idxs, axis=0)
            line_mask = tf.reduce_max(vis_masks, axis=0)

            angle_label = tf.atan(tf.tan(-vec_angle))
            angle_label = tf.reduce_max((angle_label[:,tf.newaxis, tf.newaxis]+ math.pi/2) * vis_masks , axis=0) - math.pi/2*line_mask

            thickness = 1 + (thickness-1)*2
            thickness_label = tf.reduce_max(thickness[:,tf.newaxis, tf.newaxis, tf.newaxis]*vis_masks, axis=0)

            center_vec_label = tf.map_fn(lambda x: self._gen_center_vecs_map(*x), [vecs_mask, vecs_col, vis_masks], fn_output_signature=tf.float32)
            center_vec_label = tf.reduce_sum(center_vec_label, axis=0)

            canceled_mask = tf.cast(tf.where(tf.reduce_sum(center_vec_label**2, axis=-1, keepdims=True)**0.5>tf.where(thickness_label>1, thickness_label/2, thickness_label)+1, 0., 1.), tf.float32)
            center_vec_label *= canceled_mask
            line_mask *= canceled_mask
            angle_label *= canceled_mask
            thickness_label *= canceled_mask
            
            return I, line_mask, angle_label, thickness_label, center_vec_label
        
        elif self.output_type==4:
            vis_masks, vis_idxs = self._visible_masks(masks, min_points=1)
            vis_masks = tf.concat([vis_masks, 1-tf.reduce_max(vis_masks, axis=0, keepdims=True)], axis=0)
            n = len(vis_masks)
            vis_masks = tf.pad(vis_masks, [[0, self.max_examples_num+1-n],[0,0],[0,0],[0,0]])
            return I, vis_masks
        
        elif self.output_type==5:
            mask_label, n = self._random_visible_mask(masks, min_points=1)
            vec_label = tf.reshape(gen_visible_endpoints(mask_label, vecs_mask[n], vecs_col[n], n, masks), (-1,2,2))
            vec_num = len(vec_label)
            vec_mask = tf.pad(tf.ones((vec_num,1)), [[0,self.size-vec_num], [0,0]])
            vec_label = tf.pad(vec_label, [[0,self.size-vec_num], [0,0], [0,0]])
            angle_label = tf.atan(tf.tan(-vec_angle[n]))

            points = tf.where(mask_label[...,0]>0)
            m = tf.py_function(np.random.randint, [0,len(points)], tf.int32)
            point = points[m]

            return I, mask_label, vec_label, vec_mask, angle_label, point
        
        return I, masks, vecs_col, vecs_mask, thickness
    
    def gen_full_map(self):
        args = self._gen_parameters()
        features = self._map_drawing(*args)
        return features
    
    def reload_parcel_inputs(self):
        None
    
    def new_dataset(self):
        ds = tf.data.Dataset.range(max(self.batch_size,1)*2)
        ds = ds.map(self._gen_parameters, num_parallel_calls=self.parallel_calls)
        ds = ds.map(self._map_drawing, num_parallel_calls=self.parallel_calls)
        if self.batch_size>0:
            ds = ds.batch(self.batch_size)
        ds = ds.repeat()

        return ds
    

##### ENDPOINTS MODEL ######

class GetBboxesMapMethod(tf.keras.layers.Layer):
    def __init__(self, crop_size=3, offset=3, size=32, method='bilinear', **kwargs):
        super().__init__(**kwargs)

        self.crop_size = crop_size
        self.offset = offset/2/(size-1)
        self.size = size
        self.method = method

    def build(self, input_shape):
        boxes_num = input_shape[-2]
        self.box_indices = tf.zeros((boxes_num,), dtype=tf.int32)

    def call(self, boxes, source):

        boxes = boxes/self.size
        boxes = tf.concat([boxes-self.offset, boxes+self.offset], axis=-1)

        boxes_features = tf.map_fn(lambda x: tf.image.crop_and_resize(x[0][tf.newaxis],x[1], self.box_indices, crop_size=(self.crop_size, self.crop_size), method=self.method), 
                                   [source, boxes], fn_output_signature = tf.float32)
        #boxes_features = tf.squeeze(boxes_features, axis=1)
        return boxes_features
    
class GetBboxes(tf.keras.layers.Layer):
    def __init__(self, crop_size=3, offset=3, size=32, method='bilinear', **kwargs):
        super().__init__(**kwargs)

        self.crop_size = crop_size
        self.offset = offset/2/(size-1)
        self.size = size

        self.squeeze = SqueezeImg()
        self.unsqueeze = UnSqueezeImg()
        self.method = method

        self.xy = xy_coords((size, size))[tf.newaxis]

    def call(self, boxes, source):

        B = tf.shape(boxes)[0]
        C = tf.shape(source)[-1]
        boxes_num = tf.shape(boxes)[-2]
        boxes = boxes/(self.size-1)
        boxes = tf.concat([boxes-self.offset, boxes+self.offset], axis=-1)
        #boxes = self.squeeze(boxes)
        boxes = tf.reshape(boxes, (-1,4))
        box_indices = tf.reshape(tf.repeat(tf.range(B)[:,tf.newaxis], boxes_num, axis=-1), (-1,))
        boxes_features = tf.image.crop_and_resize(source, boxes, box_indices, crop_size=(self.crop_size, self.crop_size), method=self.method)
        boxes_features = tf.reshape(boxes_features, (-1, boxes_num, self.crop_size, self.crop_size, C))
        #boxes_features = tf.reshape(boxes_features, (B, self.size, self.size, self.crop_size**2, C))
        #boxes_features = tf.reduce_max(boxes_features, axis=-2)
        return boxes_features
    
class AngleRangeLayer(tf.keras.layers.Layer):
    def __init__(self, angle_step=10, side_steps=1, **kwargs):
        super().__init__(**kwargs)

        radian_step = angle_step/180*math.pi
        self.steps = side_steps*2+1
        self.angle_shifts = tf.reshape(tf.cast(tf.linspace(-side_steps*radian_step, side_steps*radian_step, self.steps), tf.float32), (1,self.steps,1))

    def call(self, angles, ref_points):

        angles = angles[:,tf.newaxis]+self.angle_shifts
        ref_points = tf.repeat(ref_points[:,tf.newaxis], self.steps, axis=1)

        return angles, ref_points
    
class LinearSamples(tf.keras.layers.Layer):
    def __init__(self, width, points_num, sample_space=1.0, size=32, symmetric=False, **kwargs):
        super().__init__(**kwargs)

        self.size = size
        self.perp_points_width = width
        self.perp_points_num = 2*width+1
        self.sample_points_num = points_num
        self.sample_space = sample_space

        self.symmetric = symmetric

        self.perp_points_range = tf.reshape(tf.cast(tf.linspace(-self.perp_points_width, self.perp_points_width, self.perp_points_num), dtype=tf.float32), (1,1,1,self.perp_points_num,1))

        self.sample_points_range = tf.linspace(0., 1., points_num//(2 if symmetric else 1))[tf.newaxis, tf.newaxis, :, tf.newaxis]

    def build(self, input_shape):

        self.angle_dim = input_shape[1]

    def call(self, ref_angles, ref_points, thickness=None):
        ref_points = tf.expand_dims(ref_points, axis=-2)
        perp_angle = ref_angles-math.pi/2
        perp_vec = tf.cast(tf.concat([tf.sin(perp_angle), tf.cos(perp_angle)], axis=-1), tf.float32)[:,:,tf.newaxis, tf.newaxis]

        line_vec = tf.concat([tf.sin(ref_angles), tf.cos(ref_angles)], axis=-1)
        line_vec = tf.stack([-line_vec, line_vec], axis=-2)

        pos_side_mult = (self.size-1-ref_points)/line_vec
        neg_side_mult = (ref_points)/-line_vec

        vec_mult = tf.reduce_min(tf.nn.relu(pos_side_mult)+tf.nn.relu(neg_side_mult), axis=-1, keepdims=True)

        #line_vec = tf.nn.relu(line_vec)*pos_side_mult-tf.nn.relu(-line_vec)*neg_side_mult+ref_point
        line_vec = line_vec*vec_mult+ref_points

        if self.symmetric:
            left_vec, right_vec = tf.split(line_vec - ref_points, 2, axis=-2)
            left_vec = left_vec*(1-self.sample_points_range) + ref_points
            right_vec = right_vec*self.sample_points_range + ref_points

            sample_points = tf.concat([left_vec, right_vec], axis=-2)
        else:
            left_vec, right_vec = tf.split(line_vec, 2, axis=-2)
            vec = right_vec-left_vec
            sample_points = vec*self.sample_points_range + left_vec
        
        sample_space = thickness * self.sample_space if thickness is not None else self.sample_space
        
        perp_points =  self.perp_points_range * perp_vec * sample_space + sample_points[...,tf.newaxis,:]

        return perp_points, line_vec, perp_vec
    
class ExtractPointwiseEmbeddings(tf.keras.layers.Layer):
    
    def call(self, embeddings, coords, mask=None):

        coords_embs = tf.gather_nd(embeddings, tf.cast(coords, tf.int32), batch_dims=1)

        if mask is not None:
            coords_embs = coords_embs*mask

        return coords_embs
    
class Squeeze2Batch(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.out_shape = tf.concat([[-1],input_shape[2:]], axis=0)

    def call(self, inputs):
        return tf.reshape(inputs, self.out_shape)
    
class ExtractFromBatch(tf.keras.layers.Layer):
    def __init__(self, extraction_dim_size, **kwargs):
        super().__init__(**kwargs)

        self.dim = extraction_dim_size

    def build(self, input_shape):
        self.out_shape = tf.concat([[-1, self.dim],input_shape[1:]], axis=0)

    def call(self, inputs):
        return tf.reshape(inputs, self.out_shape)     
    
class EndpointsSelection(tf.keras.layers.Layer):

    def call(self, endpoints_pred, line_vec, perp_vec):
        conf, endpoints = tf.split(endpoints_pred, [1,4], axis=-1)
        _, idxs = tf.math.top_k(tf.squeeze(conf, axis=-1), k=1)
        selected_endpoints = tf.gather(endpoints, idxs, axis=1, batch_dims=1)
        selected_endpoints = tf.reshape(selected_endpoints, (-1, 2, 2))

        selected_line_vec = tf.squeeze(tf.gather(line_vec, idxs, axis=1, batch_dims=1), axis=1)
        selected_perp_vec = tf.squeeze(tf.gather(perp_vec, idxs, axis=1, batch_dims=1), axis=1)

        return selected_endpoints, selected_line_vec, selected_perp_vec
    
class VecLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1, reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(reduction=reduction,**kwargs)

        self.gamma = gamma

    def _dist(self, y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.abs(y_true-y_pred)**self.gamma, axis=-1), axis=-1)
    
    def call(self, y_true, y_pred):

        a = self._dist(y_true, y_pred)
        b = self._dist(y_true[:,::-1], y_pred)

        scores = tf.reduce_min(tf.stack([a,b], axis=-1), axis=-1)

        return scores
    

class EndpointsDecoder(tf.keras.layers.Layer):
    def __init__(self, sample_points_num, perp_points_num, sample_space=1.0, symmetric=False, normalize=True, batch_dims=1, **kwargs):
        super().__init__(**kwargs)

        self.sample_points_num = sample_points_num if not symmetric else sample_points_num//2
        self.perp_points_num = perp_points_num
        self.sample_space = sample_space
        self.symmetric = symmetric

        self.normalize = normalize
        if normalize:
            self.dim_reg = tf.constant([[sample_points_num-1,perp_points_num-1]], tf.float32)
            self.angle_reg = tf.constant([[0.,0.5]], tf.float32)

            for i in range(batch_dims):
                self.dim_reg = self.dim_reg[tf.newaxis]
                self.angle_reg = self.angle_reg[tf.newaxis]

    def call(self, endpoints, line_vec, perp_vec, ref_points=None, thickness=None):
        if self.normalize:
            endpoints_pred = endpoints/self.dim_reg - self.angle_reg
        else:
            endpoints_pred = endpoints

        if self.symmetric:
            ref_points = tf.expand_dims(ref_points, axis=-2)
            base_points =  tf.gather(endpoints_pred, [0], axis=-1)*(line_vec-ref_points) + ref_points
        else:
            left_vec, right_vec = tf.split(line_vec, 2, axis=-2)
            base_points = tf.gather(endpoints_pred, [0], axis=-1)*(right_vec-left_vec)+left_vec

        perp_width = thickness * self.sample_space * self.perp_points_num if thickness is not None else self.sample_space * self.perp_points_num
        perp_shift = tf.gather(perp_vec, 0, axis=-3)*perp_width*tf.gather(endpoints_pred, [1], axis=-1)

        decoded_endpoints = perp_shift+base_points

        return decoded_endpoints
    

class EndpointsFormatter(tf.keras.layers.Layer):
    
    def build(self, input_shape):
        self.angle_dims = input_shape[-2]

    def call(self, inputs):
        conf, endpoints = tf.split(inputs, [1,4], axis=-1)

        endpoints = tf.reshape(endpoints, (-1, self.angle_dims, 2, 2))

        conf = tf.nn.sigmoid(conf)

        return conf, endpoints
    

class GenPointMap(tf.keras.layers.Layer):
    def __init__(self, points_num, side_steps, reg_shift=0.5, reg_mult=1.0, batch_dims=1, **kwargs):
        super().__init__(**kwargs)

        self.reg_shift = reg_shift
        self.reg_mult = reg_mult

        self.points_num = points_num
        self.side_steps = side_steps

        self.xy = xy_coords((points_num, side_steps*2+1))[...,::-1]

        for i in range(batch_dims):
            self.xy = self.xy[tf.newaxis]

    def encode_coords(self, line_vec, points):
        p1 = tf.gather(line_vec, 0, axis=-2)
        p2 = tf.gather(line_vec, 1, axis=-2)
        p0 = points

        encoded_coords = tf.reduce_mean((p0-p1)/(p2-p1), axis=-1)*(self.points_num-1)
        encoded_coords = tf.stack([encoded_coords, encoded_coords*0+self.side_steps], axis=-1)

        return encoded_coords

    def call(self, line_vec, points):
        encoded_coords = self.encode_coords(line_vec, points)
        return tf.math.exp(-tf.nn.relu(tf.reduce_sum((self.xy-encoded_coords[...,tf.newaxis, tf.newaxis,:])**2, axis=-1, keepdims=True)**0.5-self.reg_shift)*self.reg_mult)
    

class GenDirectPointMap(tf.keras.layers.Layer):
    def __init__(self, reg_shift=0.5, reg_mult=1.0, **kwargs):
        super().__init__(**kwargs)

        self.reg_shift = reg_shift
        self.reg_mult = reg_mult


    def call(self, sample_points, points):
        return tf.math.exp(-tf.nn.relu(tf.reduce_sum((sample_points-points[...,tf.newaxis, tf.newaxis,:])**2, axis=-1, keepdims=True)**0.5-self.reg_shift)*self.reg_mult)


class TwoInputsMean(tf.keras.metrics.Mean):

    def update_state(self, values, *args, sample_weight=None):
        return super().update_state(values, sample_weight)

class MultiProposalVecLoss():
    def __init__(self, conf_weight=0.5, endpoint_weight=0.5, reg_shift=0.5, reg_mult=1.0, gamma=1, method='min', name='VL'):

        self.conf_weight = conf_weight
        self.endpoint_weight = endpoint_weight
        self.weights = tf.constant([[conf_weight, endpoint_weight]], dtype=tf.float32)/(conf_weight + endpoint_weight)

        self.reg_shift = reg_shift
        self.reg_mult = reg_mult
        self.gamma = gamma

        self.name = name

        self.vec_loss = VecLoss(gamma=gamma, reduction='none')
        self.method = method

        pool_funcs = {
            'min': self._min_pool,
            'weighted': self._weighted_pool,
            'best': self._best_pool
        }

        self.pool_func = pool_funcs[method]

    def get_config(self):
        return {
            'name': self.name,
            'conf_weight': self.conf_weight,
            'endpoint_weight': self.endpoint_weight,
            'reg_shift': self.reg_shift,
            'reg_mult': self.reg_mult,
            'gamma': self.gamma,
            'method': self.method
        }

    def _gen_entropy_labels(self, vec_label, line_vec):
        dists = tf.reduce_max(dist_to_line(vec_label, line_vec), axis=-2)
        labels = tf.math.exp(-tf.nn.relu(dists-self.reg_shift)*self.reg_mult)

        return labels
    
    def _min_pool(self, endpoints_losses, **kwargs):
        return tf.reduce_min(endpoints_losses, axis=-1)
    
    def _weighted_pool(self, endpoints_losses, conf_label, **kwargs):
        weights = tf.squeeze(conf_label/tf.reduce_sum(conf_label, axis=-2, keepdims=True), axis=-1)
        return tf.reduce_sum(endpoints_losses*weights, axis=-1)
    
    def _best_pool(self, endpoints_losses, conf, **kwargs):
        _, idxs = tf.math.top_k(tf.squeeze(conf, axis=-1), k=1)
        return tf.squeeze(tf.gather(endpoints_losses, idxs, axis=-1, batch_dims=1), axis=-1)
    
    def __call__(self, y_true, y_pred):
        vec_label = y_true[:,tf.newaxis]

        endpoints_pred, conf, line_vec = y_pred['endpoints_pred'], y_pred['conf'], y_pred['line_vec']

        endpoints_losses = self.vec_loss(vec_label, endpoints_pred)
        conf_label = self._gen_entropy_labels(vec_label, line_vec)

        conf_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(conf_label, conf), axis=-1)
        endpoints_loss = self.pool_func(endpoints_losses=endpoints_losses, conf_label=conf_label, conf=conf)

        loss_value = tf.reduce_sum(tf.stack([conf_loss, endpoints_loss], axis=-1)*self.weights, axis=-1)

        return (
            loss_value,
            {'vec': vec_label, 'conf': tf.squeeze(conf_label, axis=-1), 'conf_loss': conf_loss, 'endpoints_loss': endpoints_loss}, 
            {'vec': endpoints_pred, 'conf': tf.squeeze(conf, axis=-1), 'conf_loss': None, 'endpoints_loss': None}
        )




##### LINE FEATURES MODEL #####



def two_side_angle_diff(a, b, gamma=1.):
    return tf.abs(tf.sin(a-b))**gamma

def flatten(x):
    return tf.reshape(x, (-1, tf.math.reduce_prod(tf.shape(x)[1:])))

def calc_vec_angle(x):
    return tf.math.atan2(*tf.split(x, 2, axis=-1))

def sample_weighted_mean(x, sample_weight):
    x = flatten(x)
    sample_weight = flatten(sample_weight)

    sample_weight = flatten(sample_weight)
    denom = tf.reduce_sum(sample_weight, axis=-1)+1e-6
    nom = tf.reduce_sum(sample_weight*x, axis=-1)
    return nom/denom

    
class AngleLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1., name='AL', reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

        self.gamma = gamma

    def get_config(self):
        return {
            'name': self.name,
            'reduction': self.reduction,
            'gamma': self.gamma
        }
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        diffs = two_side_angle_diff(y_true, y_pred, gamma=self.gamma)
        
        if sample_weight is not None:
            loss_value = sample_weighted_mean(diffs, sample_weight)
        else:
            loss_value = tf.reduce_mean(flatten(diffs), axis=-1)

        loss_value = loss_value**(1/self.gamma)

        return tf.reduce_mean(loss_value)
    
class AngleLengthLoss(tf.keras.losses.Loss):
    def __init__(self, angle_gamma=1., length_gamma=1., dist_gamma=2., angle_weight=0.5, length_weight=0.5, dist_weight=0.5, name='AL', reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

        self.angle_gamma = angle_gamma
        self.length_gamma = length_gamma
        self.dist_gamma = dist_gamma

        weights_sum = angle_weight+length_weight+dist_weight
        self.angle_weight = angle_weight/weights_sum
        self.length_weight = length_weight/weights_sum
        self.dist_weight = dist_weight/weights_sum

    def get_config(self):
        return {
            'name': self.name,
            'reduction': self.reduction,
            'angle_gamma': self.angle_gamma,
            'length_gamma': self.length_gamma,
            'angle_weight': self.angle_weight,
            'length_weight': self.length_weight
        }
    
    @staticmethod
    def _calc_length(x):
        return tf.reduce_sum(x**2, axis=-1, keepdims=True)**0.5
    
    @staticmethod
    def _calc_dist(a,b, gamma):
        return tf.reduce_sum((a-b)**gamma, axis=-1, keepdims=True)**(1/gamma)
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        true_angle = calc_vec_angle(y_true)/2
        pred_angle = calc_vec_angle(y_pred)/2

        angle_diffs = two_side_angle_diff(true_angle, pred_angle, gamma=self.angle_gamma)

        true_length = self._calc_length(y_true)
        pred_length = self._calc_length(y_pred)

        length_diffs = tf.abs(true_length-pred_length)**self.length_gamma

        dists = self._calc_dist(y_true, y_pred, gamma=self.dist_gamma)

        if sample_weight is not None:
            angle_loss = sample_weighted_mean(angle_diffs, sample_weight)
            length_loss = sample_weighted_mean(length_diffs, sample_weight)
            dist_loss = sample_weighted_mean(dists, sample_weight)
        else:
            angle_loss = tf.reduce_mean(flatten(angle_diffs), axis=-1)
            length_loss = tf.reduce_mean(flatten(length_loss), axis=-1)
            dist_loss = tf.reduce_mean(flatten(dist_loss), axis=-1)

        angle_loss = angle_loss**(1/self.angle_gamma)
        length_loss = length_loss**(1/self.length_gamma)

        loss_value = angle_loss*self.angle_weight + length_loss*self.length_weight + dist_loss*self.dist_weight

        return tf.reduce_mean(loss_value)
    
    
class AngleActivationLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.atan(tf.tan(inputs))
    
class Vec2AngleActivationLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.math.atan2(*tf.split(inputs, 2, axis=-1))
    
class CenterVecFormatter(tf.keras.layers.Layer):

    def call(self, center_vec_feature, line_angle):
        perp_angle = line_angle-math.pi/2
        center_vec = tf.concat([tf.sin(perp_angle), tf.cos(perp_angle)], axis=-1)*center_vec_feature

        return center_vec
    

def one_side_angle_vec_points(angles, mask=None):
    ay = tf.sin(angles)
    ax = tf.cos(angles)

    ay *= tf.sign(ax)
    ax *= tf.sign(ax)

    if mask is not None:
        ay *= mask
        ax *= mask

    return tf.squeeze(ay, axis=-1), tf.squeeze(ax, axis=-1)


##### ENDPOINTS PAIRING MODEL #####

class ExtractPointwiseEmbeddings(tf.keras.layers.Layer):
    
    def call(self, embeddings, coords, mask=None):

        coords_embs = tf.gather_nd(embeddings, tf.cast(coords, tf.int32), batch_dims=1)

        if mask is not None:
            coords_embs = coords_embs*mask

        return coords_embs
    
class CrossMask(tf.keras.layers.Layer):

    def call(self, inputs):

        return tf.matmul(inputs, inputs, transpose_b=True)
    
class DiagAntiMask(tf.keras.layers.Layer):
    def __init__(self, size=96, **kwargs):
        super().__init__(**kwargs)

        self.mask = self._gen_mask(size)

    @staticmethod
    def _gen_mask(size):
        xy = xy_coords((size, size))
        return tf.cast(tf.where(xy[...,0]==xy[...,1], 0.0, 1.0), tf.float32)[tf.newaxis]
    
    def call(self, input):

        return self.mask*input
    

class RegularizedSigmoid(tf.keras.layers.Layer):

    def build(self, input_shape):
        r = len(input_shape)

        self.mult = self.add_weight('mult', shape=tf.ones((r,), dtype=tf.int32))
        self.bias = self.add_weight('bias', shape=tf.ones((r,), dtype=tf.int32))

    def call(self, inputs):

        return tf.nn.sigmoid(self.mult*inputs+self.bias)