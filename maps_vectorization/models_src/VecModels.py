import tensorflow as tf
import math
import numpy as np
from models_src.Hough import VecDrawer
from models_src.fft_lib import xy_coords
from models_src.Attn_variations import UnSqueezeImg, SqueezeImg
from models_src.UNet_model import UNet
from models_src.DETR import FFN, HeadsPermuter, MHA


def tensor_from_coords(coords, values, size=32):
    x = tf.SparseTensor(coords, values, (size,size))
    x = tf.sparse.reorder(x)
    x = tf.sparse.to_dense(x, validate_indices=False)

    return x

def one_hot_angles(x, splits, denom=2):
    bins = tf.cast(tf.round(x/(denom*math.pi)*splits) + splits//2, tf.int32)
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
        self.out_shape = (-1,)+input_shape[2:]

    def call(self, inputs):
        return tf.reshape(inputs, self.out_shape)
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        if batch_size is not None:
            batch_dim = batch_size*input_shape[1]
        else:
            batch_dim = None
        return (batch_dim, ) + input_shape[2:]
    
class ExtractFromBatch(tf.keras.layers.Layer):
    def __init__(self, extraction_dim_size, **kwargs):
        super().__init__(**kwargs)

        self.dim = extraction_dim_size

    def build(self, input_shape):
        self.out_shape = (-1, self.dim) + input_shape[1:]

    def call(self, inputs):
        return tf.reshape(inputs, self.out_shape)   

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        if batch_size is not None:
            batch_dim = batch_size//self.dim
        else:
            batch_dim = None
        return (batch_dim, self.dim) + input_shape[1:]  
    
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
        b = self._dist(y_true[...,::-1,:], y_pred)

        scores = tf.reduce_min(tf.stack([a,b], axis=-1), axis=-1)

        return scores

class BBoxVecLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1, reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(reduction=reduction,**kwargs)

        self.gamma = gamma

    def _dist(self, y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.abs(y_true-y_pred)**self.gamma, axis=-1), axis=-1)
    
    def call(self, y_true, y_pred):

        a = self._dist(y_true, y_pred)
        b = self._dist(y_true[...,::-1,:], y_pred)
        y, x = tf.split(y_true, 2, axis=-1)[::-1]
        y_true_transposed = tf.concat([y, x[...,::-1,:]], axis=-1)
        c = self._dist(y_true_transposed, y_pred)
        d = self._dist(y_true_transposed[...,::-1,:], y_pred)

        scores = tf.reduce_min(tf.stack([a,b,c,d], axis=-1), axis=-1)

        return scores
    
class MixedBBoxVecLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=1, reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(reduction=reduction,**kwargs)

        self.gamma = gamma

    def _dist(self, y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.abs(y_true-y_pred)**self.gamma, axis=-1), axis=-1)
    
    def _bbox_loss(self, y_true, y_pred):
        a = self._dist(y_true, y_pred)
        b = self._dist(y_true[...,::-1,:], y_pred)
        y, x = tf.split(y_true, 2, axis=-1)[::-1]
        y_true_transposed = tf.concat([y, x[...,::-1,:]], axis=-1)
        c = self._dist(y_true_transposed, y_pred)
        d = self._dist(y_true_transposed[...,::-1,:], y_pred)

        scores = tf.reduce_min(tf.stack([a,b,c,d], axis=-1), axis=-1)

        return scores
    
    def _vec_loss(self, y_true, y_pred):
        a = self._dist(y_true, y_pred)
        b = self._dist(y_true[...,::-1,:], y_pred)

        scores = tf.reduce_min(tf.stack([a,b], axis=-1), axis=-1)

        return scores
    
    def call(self, y_true, y_pred):

        vec_true, bbox_true = tf.split(y_true, 2, axis=-2)
        vec_pred, bbox_pred = tf.split(y_pred, 2, axis=-2)

        vec_scores = self._vec_loss(vec_true, vec_pred)
        bbox_scores = self._bbox_loss(bbox_true, bbox_pred)

        scores = vec_scores + bbox_scores
        return scores
    
class NoSplitMixedBboxVecLoss(MixedBBoxVecLoss):

    def call(self, y_true, y_pred):

        y_true = tf.stack(tf.split(y_true, 2, axis=-2), axis=-3)
        y_pred = tf.expand_dims(y_pred, axis=-3)

        scores = tf.reduce_min(self._vec_loss(y_true, y_pred), axis=-1)
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

    
'''class AngleLoss(tf.keras.losses.Loss):
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

        return tf.reduce_mean(loss_value)'''
    
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
    
    def call(self, y_true, y_pred):
        diffs = two_side_angle_diff(y_true, y_pred, gamma=self.gamma)

        return diffs
    
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
    
    def call(self, y_true, y_pred, sample_weight=None):
        true_angle = calc_vec_angle(y_true)/2
        pred_angle = calc_vec_angle(y_pred)/2

        angle_diffs = two_side_angle_diff(true_angle, pred_angle, gamma=self.angle_gamma)

        true_length = self._calc_length(y_true)
        pred_length = self._calc_length(y_pred)

        length_diffs = tf.abs(true_length-pred_length)**self.length_gamma

        dists = self._calc_dist(y_true, y_pred, gamma=self.dist_gamma)

        angle_diffs = angle_diffs**(1/self.angle_gamma)
        length_diffs = length_diffs**(1/self.length_gamma)

        loss_value = angle_diffs*self.angle_weight + length_diffs*self.length_weight + dists*self.dist_weight

        return loss_value
    
    '''def __call__(self, y_true, y_pred, sample_weight=None):
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

        return tf.reduce_mean(loss_value)'''
    
    
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
  

class PixelCrossSimilarityCrossentropy(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.0, axis=[-1,-2], name='PCS', reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

        self.squeeze_img = SqueezeImg()
        self.label_smoothing = label_smoothing
        self.ce_axis = axis

    def get_config(self):
        return {
            'name': self.name,
            'label_smoothing': self.label_smoothing,
            'axis': self.ce_axis
        }
    
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(self.squeeze_img(y_true), axis=-1)
        y_true = tf.matmul(y_true, y_true, transpose_a=True)

        loss_value = tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=self.label_smoothing, axis=self.ce_axis)

        return loss_value
  


def pixel_features_unet(input_shape, init_filters_power, levels, level_convs, init_dropout, dropout, batch_normalization, name='PxFeaturesUnet', **kwargs):
    unet_model = UNet(
        input_shape = input_shape,
        out_dims = 8,
        out_activation=None,
        init_filters_power=init_filters_power,
        levels=levels,
        level_convs=level_convs,
        color_embeddings=False,
        init_dropout=init_dropout,
        dropout=dropout,
        batch_normalization=batch_normalization,
        output_smoothing=False
    )

    inputs = unet_model.input
    out = unet_model.output

    out_shape_class, out_angle, out_thickness, out_center_vec = SplitLayer([3,2,1,2], name='Splits')(out)

    out_shape_class = tf.keras.layers.Activation('softmax', name='shape_class')(out_shape_class)
    out_angle = Vec2AngleActivationLayer(name='angle')(out_angle)
    #out_center_vec = CenterVecFormatter(name='CenterVec')(out_center_vec, out_angle)
    out_center_vec = tf.keras.layers.Identity(name='center_vec')(out_center_vec)
    out_thickness = tf.keras.layers.Identity(name='thickness')(out_thickness)
    return tf.keras.Model(inputs, {'shape_class': out_shape_class, 'angle': out_angle, 'thickness': out_thickness, 'center_vec': out_center_vec}, name=name)

class DotSimilarityLayer(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)

        self.e = epsilon

    def call(self, a, b=None):
        if b is None:
            b = a
        x = tf.matmul(a, b, transpose_b=True)
        x = tf.nn.softmax(x, axis=-1)
        max_sim = tf.reduce_max(x, axis=-1, keepdims=True) + self.e
        x = x/max_sim

        return x
    
class PixelSimilarityF1(tf.keras.metrics.Metric):
    def __init__(self, skip_first_pattern=True, threshold=0.5, name='F1', **kwargs):
        super(PixelSimilarityF1, self).__init__(name=name, **kwargs)

        self.f1 = tf.keras.metrics.F1Score(threshold=threshold, average='micro')
        self.flatten = tf.keras.layers.Flatten()
        self.score = self.add_weight(name='f1', initializer='zeros')
        self.iterations = self.add_weight(name='iters', initializer='zeros')
        self.threshold = threshold

        self.squeeze_img = SqueezeImg()
        self.squeeze_mask = SqueezeImg()
        self.skip_first = skip_first_pattern

    def get_config(self):
        return {**super().get_config(), 
                'threshold': self.threshold, 
                'skip_first_pattern': self.skip_first}

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.skip_first:
            y_true = y_true[:,1:]

        y_true = tf.squeeze(self.squeeze_img(y_true), axis=-1)
        y_true = tf.matmul(y_true, y_true, transpose_a=True)
        y_pred = y_pred*tf.reduce_max(y_true, axis=-1, keepdims=True)

        self.score.assign_add(self.f1(self.flatten(y_true), self.flatten(y_pred)))
        self.iterations.assign_add(1.0)

    def result(self):
        return self.score/self.iterations
    
class WeightedPixelCrossSimilarityCrossentropy(tf.keras.losses.Loss):
    def __init__(self, label_smoothing=0.0,  name='PCS', reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name=name, reduction=reduction, **kwargs)

        self.squeeze_img = SqueezeImg()
        self.label_smoothing = label_smoothing

    def get_config(self):
        return {
            'name': self.name,
            'label_smoothing': self.label_smoothing,
        }
    
    def call(self, y_true, y_pred):
        y_true = tf.squeeze(self.squeeze_img(y_true), axis=-1) #[B,P,HW]

        pattern_sums = tf.reduce_sum(y_true, axis=-1, keepdims=True)+1e-6 #[B,P,1]
        weights = tf.reduce_sum(y_true/pattern_sums, axis=-2) #[B,HW]
        weights_denom = tf.reduce_sum(weights, axis=-1, keepdims=True) #[B,1]
        weights = weights/weights_denom #[B,HW]

        y_true = tf.matmul(y_true, y_true, transpose_a=True) #[B,HW,HW]

        loss_value = tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=self.label_smoothing, axis=-1) #[B,HW]
        loss_value = tf.reduce_sum(loss_value*weights, axis=-1) #[B]

        return loss_value

def backbone_based_pixel_similarity_dot_model(
        backbone_args,
        backbone_generator,
        backbone_weights_path,
        backbone_trainable,
        backbone_last_layer,
        backbone_init_layer,
        color_embs_num,
        color_embs_mid_layers,
        conv_num,
        conv_dim,
        attn_dim,
        heads_num,
        use_heads,
        pre_attn_ffn_mid_layers,
        dropout,
        name='PxSimDot'
        ):
    
    conv_dim = attn_dim//2
    
    backbone_model = backbone_generator(**backbone_args)
    if backbone_weights_path is not None:
        backbone_model.load_weights(f'./{backbone_weights_path}.weights.h5')

    backbone_model.trainable = backbone_trainable
    inputs = backbone_model.input
    memory = backbone_model.get_layer(backbone_last_layer).output
    normed_img = backbone_model.get_layer(backbone_init_layer).output

    normed_img = SqueezeImg(name='Squeeze-NormedImg')(normed_img)
    color_embs_map = FFN(mid_layers=color_embs_mid_layers, mid_units=color_embs_num*2, output_units=color_embs_num, dropout=dropout, activation='relu')(normed_img)

    if conv_num>0:
        memory = tf.keras.layers.Conv2D(conv_dim, kernel_size=1, padding='same', activation='relu', name='Conv_init')(memory)
        x = memory
        for i in range(conv_num):
            x = tf.keras.layers.Conv2D(conv_dim, kernel_size=3, padding='same', activation='relu', name=f'Conv_{i+1}')(x)

        x = tf.keras.layers.Concatenate(name='Concat_Memory')([x, memory])
    else:
        x = memory
    x = SqueezeImg(name='Squeeze-Memory')(x)
    x = FFN(mid_layers=pre_attn_ffn_mid_layers, mid_units=attn_dim*2, output_units=attn_dim, dropout=dropout, activation='relu', name='features_FFN')(x)
    x = tf.keras.layers.Concatenate(name='Concat_Color_Embs')([x, color_embs_map])

    if use_heads:
        x = HeadsPermuter(num_heads=heads_num, emb_dim=(attn_dim+color_embs_num)//heads_num, name='Heads_Permute')(x)
        x = tf.keras.layers.LayerNormalization(axis=-1, name='Heads_Norm')(x)
        x = HeadsPermuter(num_heads=heads_num, emb_dim=(attn_dim+color_embs_num)//heads_num, reverse=True, name='Heads_Unpermute')(x)
    else:
        x = tf.keras.layers.LayerNormalization(axis=-1, name='Norm')(x)
    out = DotSimilarityLayer(epsilon=1e-6, name='Dot_Similarity')(x)

    return tf.keras.Model(inputs, out, name=name)

def bbox_from_vec(vec):
    p0, p2 = tf.split(vec, 2, axis=-2)
    p4 = p0
    p1 = tf.stack([p2[...,0], p0[...,1]], axis=-1)
    p3 = tf.stack([p0[...,0], p2[...,1]], axis=-1)

    bboxes = tf.concat([p0,p1,p2,p3,p4], axis=-2)

    return bboxes

def prepare_components_vecs_to_plot(components_vecs, components_class):
    vec_idxs = tf.squeeze(tf.where(components_class==2), axis=-1)
    bbox_idxs = tf.squeeze(tf.where(components_class==1), axis=-1)

    vecs = tf.gather(components_vecs, vec_idxs, axis=0)
    vecs = tf.transpose(vecs, [2,1,0])[::-1]

    bboxes = tf.gather(components_vecs, bbox_idxs, axis=0)
    bboxes = bbox_from_vec(bboxes)
    bboxes = tf.transpose(bboxes, [2,1,0])[::-1]

    return vecs, bboxes


def prepare_vec_label2plot(vec, class_idx, pred=False):

    if class_idx==1:
        #bbox
        if pred:
            vec = bbox_from_vec(vec)
        else:
            vec = tf.stack([vec[i] for i in [0,2,1,3,0]], axis=0)
    elif class_idx==2:
        #line
        vec = vec[:2]
    else:
        return None

    vec = tf.transpose(vec, [1,0])[::-1]
    return vec


def clock_radial_enc(angles, shifts_num, period=2):
    shifts = tf.reshape(tf.range(shifts_num, dtype=tf.float32), (1,1,1,shifts_num))

    encodings = tf.nn.relu(tf.cos(angles + period*shifts*math.pi/shifts_num))**(shifts_num//2)

    return encodings

def radial_dists(points, yx, size):
    return (tf.reduce_sum((yx[tf.newaxis]-points[:,tf.newaxis, tf.newaxis])**2, axis=-1, keepdims=True)**0.5)/(size*2**0.5)*math.pi

def frequency_encoding(r_embed, num_pos_features, reg=1):
    dim_t = tf.range(num_pos_features//2, dtype=tf.float32)
    dim_freq = tf.reshape(tf.stack([dim_t]*2, axis=-1), (num_pos_features,))
    dim_ph = dim_freq/(num_pos_features//2)*math.pi*2

    pos_r = r_embed/reg * dim_freq + dim_ph*reg

    encodings = tf.concat([tf.math.sin(pos_r[...,0::2]),
                        tf.math.cos(pos_r[...,1::2])], axis=-1)
    
    return encodings

def add_batch_dims(x, batch_dims):
    return tf.reshape(x, tf.concat([tf.ones((batch_dims,), dtype=tf.int32), tf.shape(x)], axis=0))


class RadialEncoding(tf.keras.layers.Layer):
    def __init__(self, emb_dim, height, width=None, inverted_angle=False, **kwargs):
        super().__init__(**kwargs)

        self.H = height
        self.W = height if width is None else width

        self.C = emb_dim
        self.inverted_angle = inverted_angle
        self.inv = -1 if inverted_angle else 1

    def build(self, input_shape):

        self.batch_dims = len(input_shape)-1

        self.yx = add_batch_dims(xy_coords((self.H, self.W))[...,::-1], self.batch_dims)

        self.shifts_num = tf.constant(self.C//2, tf.float32)
        self.radial_period = tf.constant(2, tf.float32)
        self.ring_period = tf.constant(-1, tf.float32)

        self.shifts = add_batch_dims(tf.range(self.C//2, dtype=tf.float32), self.batch_dims+2)

        self.diag = tf.constant((self.H**2 + self.W**2)**0.5, tf.float32)

        self.radial_reg = tf.constant(self.C//4, tf.float32)

    def calc_angles(self, sample_points):
        return tf.math.atan2(*tf.split((self.yx-sample_points[...,tf.newaxis, tf.newaxis,:])*self.inv, 2, axis=-1))
    
    def clock_radial_enc(self, angles, period):
        return tf.nn.relu(tf.cos(angles + period*self.shifts*math.pi/self.shifts_num))**self.radial_reg
    
    def radial_dists(self, sample_points):
        return (tf.reduce_sum((self.yx-sample_points[...,tf.newaxis, tf.newaxis, :])**2, axis=-1, keepdims=True)**0.5)/self.diag*math.pi
    
    def freq_variables(self):
        dim_t = tf.range(self.radial_reg, dtype=tf.float32)
        dim_freq = tf.reshape(tf.stack([dim_t]*2, axis=-1), (self.shifts_num,))
        dim_ph = dim_freq/(self.radial_reg)*math.pi*2

        return dim_freq, dim_ph
    
    def frequency_encoding(self, r_embed):

        pos_r = r_embed * self.dim_freq + self.dim_ph

        encodings = tf.concat([tf.math.sin(pos_r[...,0::2]),
                            tf.math.cos(pos_r[...,1::2])], axis=-1)
        
        return encodings
    
    def compute_output_shape(self, input_shape):
        return input_shape[:self.batch_dims]+(self.H, self.W, self.C)
    

class SeparateRadialEncoding(RadialEncoding):

    def call(self, sample_points):

        angles = self.calc_angles(sample_points)
        dists = self.radial_dists(sample_points)

        angle_encodings = self.clock_radial_enc(angles, self.radial_period)
        ring_encodings = self.clock_radial_enc(dists, self.ring_period)
        
        encodings = tf.concat([angle_encodings, ring_encodings], axis=-1)

        return encodings
    
class FrequencyRadialEncoding(RadialEncoding):

    def build(self, input_shape):
        super().build(input_shape)

        self.dim_freq, self.dim_ph = self.freq_variables()

    def call(self, sample_points):
        
        angles = self.calc_angles(sample_points) 
        if not self.inverted_angle:
            angles += (math.pi)
        dists = self.radial_dists(sample_points)*2

        angle_encodings = self.frequency_encoding(angles)
        ring_encodings = self.frequency_encoding(dists)

        encodings = tf.concat([angle_encodings, ring_encodings], axis=-1)

        return encodings

class ExtractSampleLayer(tf.keras.layers.Layer):
    def __init__(self, batch_dims=1, **kwargs):
        super().__init__(**kwargs)

        self.batch_dims = batch_dims

    def call(self, source, idxs):
        return tf.gather_nd(source, tf.cast(idxs, tf.int32), batch_dims=self.batch_dims)

class DetectionMHA(MHA):
    def __init__(self, value_pos_enc=True, key_pos_enc=False, query_pos_enc=False, pos_enc_matmul=False, return_weights=True, return_scores=False, **kwargs):
        super().__init__(**kwargs)

        self.key_pos_enc = key_pos_enc
        self.value_pos_enc = value_pos_enc
        self.query_pos_enc = query_pos_enc
        self.pos_enc_matmul = pos_enc_matmul

        self.return_weights = return_weights
        self.return_scores = return_scores

        if pos_enc_matmul:
            self.pe_d = tf.keras.layers.Dense(kwargs['output_dim'])
            self.pe_output_perm = HeadsPermuter(kwargs['num_heads'], reverse=True)

    def call(self, V, Q, K, pos_enc, mask=None, query_pos_enc=None):
        Q = self.Q_head_extractior(self.Q_d(Q))
        K = self.K_head_extractior(self.K_d(K))

        pos_enc = tf.expand_dims(pos_enc, axis=-3)

        if self.query_pos_enc:
            Q += pos_enc if query_pos_enc is None else tf.expand_dims(query_pos_enc, axis=-3)
            
        if self.key_pos_enc:
            K += pos_enc

        scores = tf.matmul(Q, K, transpose_b=True)/self.denominator
        if mask is not None:
            if self.soft_mask:
                scores = scores + mask
            else:
                cross_mask = tf.expand_dims(tf.matmul(mask, mask, transpose_b=True), axis=1)
                scores = scores+((cross_mask-1)*math.inf)
        weights = self.softmax(scores, axis=self.softmax_axis)

        V = self.V_head_extractior(self.V_d(V))
        if self.value_pos_enc:
            V += pos_enc
        if not self.T:
            V = tf.matmul(weights, V)
        else:
            V *= tf.transpose(weights, perm=[0,1,3,2])

        V = self.O_d(self.output_perm(V))

        if self.pos_enc_matmul:
            V += self.pe_output_perm(self.pe_d(tf.matmul(weights, pos_enc)))

        if (mask is not None) & (self.soft_mask==False):
            return V*mask
        
        if (not self.return_scores) & (not self.return_weights):
            return V
        
        outputs = [V]

        if self.return_scores:
            outputs.append(scores)
        if self.return_weights:
            outputs.append(weights)
        return outputs
    
class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)

        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
    

### RADIAL ENC VEC MODEL
    

class SplitLayer(tf.keras.layers.Layer):
    def __init__(self, splits, axis=-1, **kwargs):
        super().__init__(**kwargs)

        self.splits = splits
        self.axis = axis

    def call(self, inputs):
        return tf.split(inputs, self.splits, self.axis)
    
    @staticmethod
    def _is_iter(x):
        return hasattr(x, '__iter__')
    
    def non_iter_output_shape(self, shape, axis):
        splitted_shape = tuple([d if (i!=axis) | (d is None) else d//self.splits for i,d in enumerate(shape)])
        return tuple([splitted_shape for _ in range(self.splits)])

    def iter_output_shape(self, shape, axis):
        return tuple([tuple([d if (i!=axis) | (d is None) else s for i,d in enumerate(shape)]) for s in self.splits])
    
    def compute_output_shape(self, input_shape):
        axis = self.axis if self.axis>0 else len(input_shape)+self.axis

        if self._is_iter(self.splits):
            output_shape = self.iter_output_shape(input_shape, axis)
        else:
            output_shape = self.non_iter_output_shape(input_shape, axis)

        return output_shape
    
class VecClassSplit(tf.keras.layers.Layer):

    def build(self, input_shape):
        x_shape, splits_shape = input_shape
        self.extra_dims = len(x_shape)-len(splits_shape)


    def call(self, inputs):
        x, splits = inputs[0], inputs[1]
        #splits = splits[...,tf.newaxis, tf.newaxis]
        splits = tf.reshape(splits, tf.concat([tf.shape(splits), tf.ones((self.extra_dims,), tf.int32)], axis=0))
        return tf.concat([x*splits, x*(1-splits)], axis=-2)
    
    def compute_output_shape(self, input_shape):
        x_shape = input_shape[0]
        axis = len(x_shape)-2
        return tuple([d if (i!=axis) | (d is None) else d*2 for i,d in enumerate(x_shape)])
    
class AddNorm(tf.keras.layers.Layer):
    def __init__(self, norm_axis=-1, **kwargs):
        super().__init__(**kwargs)

        self.norm = tf.keras.layers.LayerNormalization(axis=norm_axis)

    def call(self, inputs, training=None):
        a, b = inputs[0], inputs[1]
        return self.norm(a+b, training=training)
    
    def compute_output_shape(self, input_shape):
        return [None if any([x is None for x in [a,b]]) else max(a,b) for a,b in zip(*input_shape)]
    
class SubtractNorm(AddNorm):

    def call(self, inputs):
        a, b = inputs[0], inputs[1]
        return self.norm(a-b)
    
class AngleLengthVecDecoder(tf.keras.layers.Layer):
    def __init__(self, exp_activation=True, **kwargs):
        super().__init__(**kwargs)

        self.exp_activation = exp_activation

    def call(self, inputs):
        a1, a2, len1, len2 = tf.split(inputs, 4, axis=-1)
        angle = tf.math.atan2(tf.nn.tanh(a1), tf.nn.tanh(a2))
        if self.exp_activation:
            len1, len2 = tf.math.exp(len1), tf.math.exp(len2)

        p1 = tf.concat([tf.sin(angle), tf.cos(angle)], axis=-1)*len1
        p2 = tf.concat([tf.sin(angle-math.pi), tf.cos(angle-math.pi)], axis=-1)*len2

        vec = tf.concat([p1,p2], axis=-2)

        return vec
    
class SampleRadialSearchHead(tf.keras.Model):
    def __init__(self, num_samples, ffn_mid_layers, mid_units, activation, dropout=0.0, angle_pred=False, exp_activation=True, return_logits=False, thickness_pred=False, raw_vecs=False, **kwargs):
        super().__init__(**kwargs)

        splits = [4,3]
        if thickness_pred:
            splits.append(1)

        self.ffn = FFN(mid_layers=ffn_mid_layers, mid_units=mid_units, output_units=sum(splits), dropout=dropout, activation=activation, name=f'{self.name}-Vec-Pred-FFN')
        self.split = SplitLayer(splits=splits, axis=-1, name=f'{self.name}-Vec-Class-Split')
        self.squeeze_class = tf.keras.layers.Reshape((num_samples,3), name=f'{self.name}-Class-Pred-Squeeze')
        #self.class_sigmoid = tf.keras.layers.Activation('sigmoid', name=f'{self.name}-Class-Output')
        self.class_softmax = tf.keras.layers.Softmax(axis=-1, name=f'{self.name}-Class-Output')

        if not angle_pred:
            self.vec_reshape = tf.keras.layers.Reshape((num_samples, 2,2), name=f'{self.name}-Out-Vec-Formatter')
        else:
            self.vec_reshape = AngleLengthVecDecoder(exp_activation=exp_activation, name=f'{self.name}-Out-Vec-Formatter')

        self.sample_reshape = tf.keras.layers.Reshape((num_samples, 1,2), name=f'{self.name}Out-Samples-Reshape')
        self.add = tf.keras.layers.Add(name=f'{self.name}-Coords-Add')

        self.vecbbox_split = VecClassSplit(name=f'{self.name}-Vecs-Output')

        self.return_logits = return_logits
        self.thickness_pred = thickness_pred
        self.raw_vecs = raw_vecs

        if thickness_pred:
            self.squeeze_thickness = tf.keras.layers.Reshape((num_samples,1), name=f'{self.name}-Thickness-Pred-Squeeze')

    def call(self, sample_features, sample_coords, split_mask=None, training=None):

        x = self.ffn(sample_features, training=training)
        pred_elems = self.split(x)
        x, class_pred = pred_elems[:2]
        #class_pred = self.class_sigmoid(self.squeeze_class(class_pred))
        class_logits = self.squeeze_class(class_pred)
        class_pred = self.class_softmax(class_logits)

        x = self.vec_reshape(x)
        sample_coords = self.sample_reshape(sample_coords)

        x = self.add([x, sample_coords])

        if split_mask is None:
            splited_vecs = x
        else:
            splited_vecs = self.vecbbox_split([x, split_mask])

        output_elems = [splited_vecs, class_pred]

        if self.raw_vecs:
            output_elems.append(x)

        if self.thickness_pred:
            output_elems.append(self.squeeze_thickness(pred_elems[2]))

        if self.return_logits:
            output_elems.append(class_logits)

        return output_elems
    
class SampleRadialSearchFeaturesExtraction(tf.keras.Model):
    def __init__(self, embs_dim, color_embs_dim, mid_layers, activation, dropout=0.0, batch_dims=1, memory_input=True,**kwargs):
        super().__init__(**kwargs)


        features_embs_dim = embs_dim-color_embs_dim
        self.memory_input = memory_input
        if memory_input:
            self.f_ffn = FFN(mid_layers=mid_layers, mid_units=features_embs_dim*2, output_units=features_embs_dim, dropout=dropout, activation=activation, name=f'{self.name}-Memory-FFN')
            self.f_concat = tf.keras.layers.Concatenate(name=f'{self.name}-Memory-Color-Concat')
            self.fc_ffn = FFN(mid_layers=mid_layers, mid_units=embs_dim, output_units=embs_dim, dropout=dropout, activation=activation, name=f'{self.name}-Output-Embeddings')
            
        self.c_ffn = FFN(mid_layers=mid_layers, mid_units=color_embs_dim*2, output_units=color_embs_dim, dropout=dropout, activation=activation, name=f'{self.name}-ColorEmbs-FFN')

        self.extract_samples = ExtractSampleLayer(batch_dims=batch_dims, name=f'{self.name}-Extract-Sample')
        self.expand_samples = ExpandDimsLayer(axis=-2, name=f'{self.name}-Expand-Samples')
        self.squeeze_features = SqueezeImg(name=f'{self.name}-Squeeze-Features')
        self.expand_features = ExpandDimsLayer(axis=-3, name=f'{self.name}-Expand-Features')
        self.squeeze_enc = SqueezeImg(name=f'{self.name}-Squeeze-PosEnc')


    def call(self, sample_inputs, memory, normed_img, pos_enc, training=None):

        if self.memory_input:
            features = self.f_ffn(memory, training=training)
            color_features = self.c_ffn(normed_img, training=training)
            features = self.fc_ffn(self.f_concat([features, color_features]), training=training)
        else:
            features = self.c_ffn(normed_img, training=training)

        sample_features = self.expand_samples(self.extract_samples(features, sample_inputs))

        features = self.expand_features(self.squeeze_features(features))
        pos_enc = self.squeeze_enc(pos_enc)

        return features, sample_features, pos_enc
    

class SampleRadialEncoding(RadialEncoding):
    def __init__(self, expand_a=True, expand_b=True, inverted_angle=False, **kwargs):
        super().__init__(**kwargs)

        self.expand_a = expand_a
        self.expand_b = expand_b
        self.inverted_angle = inverted_angle
    
    def calc_angles(self, a, b):

        return tf.math.atan2(*tf.split((tf.expand_dims(b, axis=-3) if self.expand_b else b)-(tf.expand_dims(a, axis=-2) if self.expand_a else a), 2, axis=-1))
    
    def clock_radial_enc(self, angles, period):
        return tf.nn.relu(tf.cos(angles + period*self.shifts*math.pi/self.shifts_num))**self.radial_reg
    
    def radial_dists(self, a, b):
        return (tf.reduce_sum(((tf.expand_dims(b, axis=-3) if self.expand_b else b)-(tf.expand_dims(a, axis=-2) if self.expand_a else a))**2, axis=-1, keepdims=True)**0.5)/self.diag*math.pi
    
class SampleSeparateRadialEncoding(SampleRadialEncoding):

    def call(self, a, b):

        angles = self.calc_angles(a,b)
        dists = self.radial_dists(a,b)

        angle_encodings = self.clock_radial_enc(angles, self.radial_period)
        ring_encodings = self.clock_radial_enc(dists, self.ring_period)
        
        encodings = tf.concat([angle_encodings, ring_encodings], axis=-1)

        return encodings
    
class SampleFrequencyRadialEncoding(SampleRadialEncoding):

    def build(self, input_shape):
        super().build(input_shape)

        self.dim_freq, self.dim_ph = self.freq_variables()

    def call(self, a, b):
        
        angles = self.calc_angles(a, b) 
        if not self.inverted_angle:
            angles += (math.pi)
        dists = self.radial_dists(a, b)*2

        angle_encodings = self.frequency_encoding(angles)
        ring_encodings = self.frequency_encoding(dists)

        encodings = tf.concat([angle_encodings, ring_encodings], axis=-1)

        return encodings
    
class ExpandedQueriesMHA(MHA):

    def call(self, V, Q, K, pos_enc):

        Q = self.Q_head_extractior(self.Q_d(Q))
        K = self.K_head_extractior(self.K_d(K))

        Q += tf.expand_dims(pos_enc, axis=-3)

        scores = tf.reduce_sum(Q*K, axis=-1)/self.denominator
        weights = tf.expand_dims(tf.nn.softmax(scores, axis=-1), axis=-2)
        
        V = self.V_head_extractior(self.V_d(V))
        V = tf.matmul(weights, V)

        V = self.O_d(self.output_perm(V))

        return V, weights
    
class QuerySamplingLayer(tf.keras.layers.Layer):
    def __init__(self, queries_num, mid_layers, mid_units, activation, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.ffn = FFN(mid_layers=mid_layers, mid_units=mid_units, output_units=queries_num*2, dropout=dropout, activation=activation)
        self.queries_num = queries_num

    def build(self, input_shape):
        self.sample_points_num = input_shape[0][-3]
        
    def call(self, inputs, training=None):

        sample_features, sample_points = inputs[0], inputs[1]

        query_points = self.ffn(sample_features, training=training)
        query_points = tf.reshape(query_points, (-1, self.sample_points_num, self.queries_num, 2))
        query_points += tf.expand_dims(sample_points, axis=-2)

        return query_points
    
class SampleQueryExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, cut_off=1, gamma=2, method='bilinear', **kwargs):
        super().__init__(**kwargs)

        self.cut_off = cut_off
        self.gamma = gamma
        self.method = method

    def build(self, input_shape):
        f_shape, q_shape = input_shape[0], input_shape[1]
        
        self.size = int(f_shape[-2]**0.5)
        self.C = f_shape[-1]

        self.sample_points_num = q_shape[-3]
        self.query_points_num = q_shape[-2]

        self.yx = tf.reshape(xy_coords((self.size, self.size))[...,::-1], (1,1,1,self.size**2,2))

    def _calc_interpolation_mask(self, p):
        diffs = tf.reduce_sum(tf.abs(p-self.yx), axis=-1, keepdims=False)
        interpolation_mask = tf.nn.relu(1-diffs+1e-4)**self.gamma
        interpolation_mask /= tf.reduce_sum(interpolation_mask, axis=-1, keepdims=True)+1e-4

        return interpolation_mask

    def call(self, inputs):
        features, query_points = inputs[0], inputs[1]

        interpolation_mask = self._calc_interpolation_mask(tf.expand_dims(query_points, axis=-2))
        query_samples = tf.matmul(interpolation_mask, features)

        return query_samples
    

class SampleQueryMessagePassing(tf.keras.layers.Layer):
    def __init__(self, mid_layers, mid_units, activation, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.mid_layers = mid_layers
        self.mid_units = mid_units
        self.activation = activation
        self.dropout = dropout

        self.message_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.out_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def build(self, input_shape):
        C = input_shape[-1]

        self.ffn = FFN(mid_layers=self.mid_layers, mid_units=self.mid_units, output_units=C, dropout=self.dropout, activation=self.activation)

    def call(self, sample_features, query_samples, pos_enc, training=None):

        query_samples += pos_enc

        message = tf.nn.sigmoid(self.ffn(tf.concat([sample_features - query_samples, query_samples], axis=-1), training=training))

        message = self.message_norm(tf.reduce_sum(message, axis=-2, keepdims=True), training=training)

        output = self.out_norm(sample_features+message, training=training)

        return output
    

class RadialSearchFeaturesExtraction(tf.keras.Model):
    def __init__(self, embs_dim, color_embs_dim, mid_layers, activation, dropout=0.0, batch_dims=1, **kwargs):
        super().__init__(**kwargs)

        self.only_colors = embs_dim==color_embs_dim

        features_embs_dim = embs_dim-color_embs_dim
        self.c_ffn = FFN(mid_layers=mid_layers, mid_units=color_embs_dim*2, output_units=color_embs_dim, dropout=dropout, activation=activation, name=f'{self.name}-ColorEmbs-FFN')

        if not self.only_colors:
            self.f_ffn = FFN(mid_layers=mid_layers, mid_units=features_embs_dim*2, output_units=features_embs_dim, dropout=dropout, activation=activation, name=f'{self.name}-Memory-FFN')
            self.f_concat = tf.keras.layers.Concatenate(name=f'{self.name}-Memory-Color-Concat')
            self.fc_ffn = FFN(mid_layers=mid_layers, mid_units=embs_dim, output_units=embs_dim, dropout=dropout, activation=activation, name=f'{self.name}-Output-Embeddings')

        self.squeeze_features = SqueezeImg(name=f'{self.name}-Squeeze-Features')
        #self.expand_features = vcm.ExpandDimsLayer(axis=-2, name=f'{self.name}-Expand-Features')
        self.squeeze_enc = SqueezeImg(name=f'{self.name}-Squeeze-PosEnc')



    def call(self, memory, normed_img, pos_enc, training=None):

        if not self.only_colors:
            color_features = self.c_ffn(normed_img, training=training)
            features = self.f_ffn(memory, training=training)

            features = self.squeeze_features(self.fc_ffn(self.f_concat([features, color_features]), training=training))
            #features = self.expand_features(self.squeeze_features(features))
        else:
            features = self.squeeze_features(self.c_ffn(normed_img, training=training))
        
        pos_enc = self.squeeze_enc(pos_enc)

        return features, pos_enc
    
class YXcoordsLayer(tf.keras.layers.Layer):
    def __init__(self, size, squeeze_output=True, **kwargs):
        super().__init__(**kwargs)

        self.size = size
        self.squeeze_output = squeeze_output

        self.yx = xy_coords(size)[tf.newaxis,...,::-1]
        if self.squeeze_output:
            self.yx = tf.reshape(self.yx, (1,size[0]*size[1], 2))

    def call(self, inputs=None):
        return self.yx
    
class SelfRadialMHA(MHA):

    def call(self, pos_enc, Q, K):

        Q = self.Q_head_extractior(self.Q_d(Q))
        K = self.K_head_extractior(self.K_d(K))

        scores = tf.matmul(Q, K, transpose_b=True)/self.denominator
        weights = tf.nn.softmax(scores, axis=-1)
        
        V = tf.expand_dims(pos_enc, axis=-4)
        V = tf.squeeze(tf.matmul(tf.expand_dims(weights, axis=-2), V), axis=-2)

        V = self.O_d(self.output_perm(V))

        return V, weights
    
def radial_enc_pixel_features_model_generator(
        enc_type,
        num_heads,
        embs_dim,
        color_embs_dim,
        size,
        embs_mid_layers,
        dropout,
        activation,
        out_mid_layers,
        attns_num,
        concat_memory,
        progressive,
        backbone_args,
        backbone_weights_path,
        backbone_generator,
        backbone_last_layer,
        backbone_init_layer,
        backbone_trainable,
        inverted_angle,
        name='PxFeaturesRadEnc'
):
    
    colors_only = embs_dim==color_embs_dim

    if colors_only:
        memory=None
        img_inputs = tf.keras.layers.Input((size, size, 3))
        normed_img = tf.keras.layers.BatchNormalization(name='Batch-Normalization')(img_inputs)
    else:
        backbone_model = backbone_generator(**backbone_args)
        if backbone_weights_path is not None:
            backbone_model.load_weights(f'./{backbone_weights_path}.weights.h5')

        backbone_model.trainable = backbone_trainable

        img_inputs = backbone_model.input
        memory = backbone_model.get_layer(backbone_last_layer).output
        normed_img = backbone_model.get_layer(backbone_init_layer).output

    
    #########
    enc_func = FrequencyRadialEncoding if enc_type!='separate' else SeparateRadialEncoding
    enc_label = 'Freq' if enc_type!='separate' else 'Sep'

    coords = YXcoordsLayer(size=(size,size), squeeze_output=True, name='Img-Coords')()

    pos_enc = enc_func(emb_dim=embs_dim//num_heads, height=size, inverted_angle=inverted_angle, name=f'{enc_label}RadialEncoding')(coords)

    features, pos_enc = RadialSearchFeaturesExtraction(embs_dim=embs_dim, 
                                                        color_embs_dim=color_embs_dim, 
                                                        mid_layers=embs_mid_layers,
                                                        activation=activation,
                                                        dropout=dropout,
                                                        batch_dims=1,
                                                        name='RSFE')(memory, normed_img, pos_enc)

    print(pos_enc.shape, features.shape)

    for i in range(attns_num):
        #V = tf.keras.layers.Permute([2,1,3], name=f'PreMHA-Permute_{i+1}')(features)
        i_heads = num_heads*2**i if progressive else num_heads
        i_embs = embs_dim*2**i if progressive else embs_dim

        x, _ = SelfRadialMHA(output_dim=i_embs, value_dim=i_embs, key_dim=i_embs, num_heads=i_heads, name=f'MHA_{i+1}')(pos_enc, features, features)
        #print(x.shape)
        if i>0:
            if progressive:
                features = FFN(mid_layers=out_mid_layers, mid_units=i_embs, output_units=i_embs, dropout=dropout, activation=activation, name=f'Progressive-SkipCon-FFN_{i+1}')(features)
            features = AddNorm(norm_axis=-1, name=f'PostMHA-AddNorm_{i+1}')([features, x])
            x = FFN(mid_layers=out_mid_layers, mid_units=i_embs*2, output_units=i_embs, dropout=0.0, activation=activation, name=f'Decoder-FFN_{i+1}')(features)
            features = AddNorm(norm_axis=-1, name=f'PostFFN-AddNorm_{i+1}')([features, x])
        else:
            features = FFN(mid_layers=out_mid_layers, mid_units=i_embs*2, output_units=i_embs, dropout=0.0, activation=activation, name=f'Decoder-FFN_{i+1}')(x)
    
    if concat_memory:
        memory = SqueezeImg(name='Squeeze-Memory')(memory)
        features = tf.keras.layers.Concatenate(axis=-1, name='Concat-Memory')([memory, features])
        
    out = FFN(mid_layers=out_mid_layers, mid_units=i_embs*2, output_units=8, dropout=dropout, activation=activation, name=f'Out-FFN')(features)
    out = tf.keras.layers.Reshape((size, size, 8))(out)

    out_shape_class, out_angle, out_thickness, out_center_vec = SplitLayer([3,2,1,2], name='Splits')(out)

    out_shape_class = tf.keras.layers.Activation('softmax', name='shape_class')(out_shape_class)
    out_angle = Vec2AngleActivationLayer(name='angle')(out_angle)
    out_center_vec = tf.keras.layers.Identity(name='center_vec')(out_center_vec)
    out_thickness = tf.keras.layers.Identity(name='thickness')(out_thickness)
    model = tf.keras.Model(img_inputs, {'shape_class': out_shape_class, 'angle': out_angle, 'thickness': out_thickness, 'center_vec': out_center_vec}, name=name)
    
    return model

class IntegralVecRadialEncodingPreprcessing(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.N = input_shape[1]

    @staticmethod
    def calc_vec_angle(x):
        return tf.math.atan2(*tf.split(tf.squeeze(tf.math.subtract(*tf.split(x, 2, axis=-2)), axis=-2), 2, axis=-1))
    
    @staticmethod
    def get_rotation_matrix(x):
        return tf.stack([tf.stack([tf.cos(x), -tf.sin(x)], axis=-1),tf.stack([tf.sin(x), tf.cos(x)], axis=-1)], axis=-2)
    
    @staticmethod
    def line_thickness_aplication(by1, by2, bx1, bx2, thickness):
        shift = thickness/2
        by1 -= shift
        by2 += shift

        return tf.stack([by1, bx1, by2, bx2], axis=-2)
    
    @staticmethod
    def shape_thickness_aplication(rel_vecs):
        y1x1 = tf.reduce_min(rel_vecs, axis=-3, keepdims=False)
        y2x2 = tf.reduce_max(rel_vecs, axis=-3, keepdims=False)
        return tf.concat([y1x1, y2x2], axis=-2)
    
    
    def call(self, sample_points, vec_pred, thickness_pred, class_pred_logit, training=None):

        vec_angle = self.calc_vec_angle(vec_pred)

        rot_matrix = tf.expand_dims(self.get_rotation_matrix(vec_angle), axis=1)

        rel_vecs = tf.expand_dims(tf.expand_dims(vec_pred, axis=1)-tf.expand_dims(tf.expand_dims(sample_points, axis=2), axis=3), axis=-1)

        rot_vecs = tf.squeeze(tf.matmul(rot_matrix, rel_vecs), axis=-1)
        r = tf.expand_dims(vec_angle+math.pi, axis=1)

        by1, bx2, by2, bx1 = tf.split(tf.reshape(rot_vecs, (-1,self.N, self.N, 4)), 4, axis=-1)
        thickness_pred = tf.expand_dims(thickness_pred, axis=1)

        line_rel_coords = self.line_thickness_aplication(by1, by2, bx1, bx2, thickness_pred)
        shape_rel_coords = self.shape_thickness_aplication(rel_vecs)

        class_probs = tf.expand_dims(tf.nn.softmax(class_pred_logit[...,1:], axis=-1), axis=1)

        return r, line_rel_coords, shape_rel_coords, class_probs
    

class IntegralVecRadialEncoding(tf.keras.layers.Layer):
    def __init__(self, emb_dim, riemann_samples, size, **kwargs):
        super().__init__(**kwargs)

        self.C = emb_dim
        self.c = emb_dim//4

        self.n = riemann_samples
        self.s = size*(2**0.5)

        self.k = tf.cast(tf.reshape(tf.range(riemann_samples)+1, (1,1,1,1,riemann_samples)), tf.float32)
        self.freq = tf.cast(tf.reshape(tf.range(self.c), (1,1,1,self.c,1)), tf.float32)

    def calc_vec_arg_riemanns(self, b1, b2):
        return self.k*(-b1+b2)/self.n + b1 - (-b1+b2)/(2*self.n)
    
    def calc_angle_angle_arg(self, y_vec_arg, x_vec_arg, r):
        return self.freq*(r + tf.math.atan2(y_vec_arg, x_vec_arg)) + 2*math.pi*self.freq/self.c
    
    def calc_ring_angle_arg(self, y_vec_arg, x_vec_arg):
        return self.freq*math.pi*((y_vec_arg**2 + x_vec_arg**2)**0.5)/self.s + 2*math.pi*self.freq/self.c
    
    def calc_prefix_arg(self, by1, by2, bx1, bx2):
        return tf.squeeze((-by1+by2)*(-bx1+bx2)/self.n, axis=-1)
    
    def calc_vec_integral(self, prefix_arg, angle_arg, func):
        return prefix_arg*tf.reduce_sum(func(angle_arg), axis=-1)
        
    def call(self, r, rel_coords):
        if r is not None:
            r = tf.expand_dims(r, axis=-1)
        else:
            r = math.pi

        by1, bx1, by2, bx2 = tf.split(rel_coords, 4, axis=-2)

        y_vec_arg = self.calc_vec_arg_riemanns(by1, by2)
        x_vec_arg = self.calc_vec_arg_riemanns(bx1, bx2)
        prefix_arg = self.calc_prefix_arg(by1, by2, bx1, bx2)

        angle_angle_arg = self.calc_angle_angle_arg(y_vec_arg, x_vec_arg, r)
        ring_angle_arg = self.calc_ring_angle_arg(y_vec_arg, x_vec_arg)

        sin_angle = self.calc_vec_integral(prefix_arg, angle_angle_arg, tf.sin)
        cos_angle = self.calc_vec_integral(prefix_arg, angle_angle_arg, tf.cos)
        sin_ring = self.calc_vec_integral(prefix_arg, ring_angle_arg, tf.sin)
        cos_ring = self.calc_vec_integral(prefix_arg, ring_angle_arg, tf.cos)

        return tf.concat([sin_angle, cos_angle, sin_ring, cos_ring], axis=-1)
    
class IntegralVecRadialEncoding(tf.keras.layers.Layer):
    def __init__(self, emb_dim, riemann_samples, size, **kwargs):
        super().__init__(**kwargs)

        self.C = emb_dim
        self.c = emb_dim//4

        self.n = riemann_samples
        self.s = size*(2**0.5)

        self.k = tf.cast(tf.reshape(tf.range(riemann_samples)+1, (1,1,1,1,riemann_samples)), tf.float32)
        self.freq = tf.cast(tf.reshape(tf.range(self.c), (1,1,1,self.c,1)), tf.float32)

    def calc_vec_arg_riemanns(self, b1, b2):
        return self.k*(-b1+b2)/self.n + b1 - (-b1+b2)/(2*self.n)
    
    def calc_angle_angle_arg(self, y_vec_arg, x_vec_arg, r):
        return self.freq*(r + tf.math.atan2(y_vec_arg, x_vec_arg)) + 2*math.pi*self.freq/self.c
    
    def calc_ring_angle_arg(self, y_vec_arg, x_vec_arg):
        return self.freq*math.pi*((y_vec_arg**2 + x_vec_arg**2)**0.5)/self.s + 2*math.pi*self.freq/self.c
    
    def calc_prefix_arg(self, by1, by2, bx1, bx2):
        return tf.squeeze((-by1+by2)*(-bx1+bx2)/self.n, axis=-1)
    
    def calc_vec_integral(self, prefix_arg, angle_arg, func):
        return prefix_arg*tf.reduce_sum(func(angle_arg), axis=-1)
    
    @staticmethod
    def calc_field(coords):
        y1, x1, y2, x2 = tf.split(tf.squeeze(coords, axis=-1), 4, axis=-1)
        return (x2-x1)*(y2-y1)
        
    def call(self, r, rel_coords):
        if r is not None:
            r = tf.expand_dims(r, axis=-1)
        else:
            r = math.pi

        by1, bx1, by2, bx2 = tf.split(rel_coords, 4, axis=-2)

        y_vec_arg = self.calc_vec_arg_riemanns(by1, by2)
        x_vec_arg = self.calc_vec_arg_riemanns(bx1, bx2)
        prefix_arg = self.calc_prefix_arg(by1, by2, bx1, bx2)

        angle_angle_arg = self.calc_angle_angle_arg(y_vec_arg, x_vec_arg, r)
        ring_angle_arg = self.calc_ring_angle_arg(y_vec_arg, x_vec_arg)

        sin_angle = self.calc_vec_integral(prefix_arg, angle_angle_arg, tf.sin)
        cos_angle = self.calc_vec_integral(prefix_arg, angle_angle_arg, tf.cos)
        sin_ring = self.calc_vec_integral(prefix_arg, ring_angle_arg, tf.sin)
        cos_ring = self.calc_vec_integral(prefix_arg, ring_angle_arg, tf.cos)

        field = self.calc_field(rel_coords)+1e-6

        return tf.concat([sin_angle, cos_angle, sin_ring, cos_ring], axis=-1)/field
    
class LineShapeEncodingConcatenation(tf.keras.layers.Layer):

    def call(self, class_probs, line_encoding, shape_encoding):

        return tf.squeeze(tf.matmul(tf.stack([shape_encoding, line_encoding], axis=-1), tf.expand_dims(class_probs, axis=-1)), axis=-1)
    
class BinarisedSoftmax(tf.keras.layers.Layer):
    def __init__(self, axis=-1, reg=1e2, **kwargs):
        super().__init__(**kwargs)

        self.axis = axis
        self.reg = reg

    def call(self, inputs):
        x = inputs-tf.reduce_max(inputs, axis=self.axis, keepdims=True)
        x = tf.nn.tanh(x*self.reg)
        x = tf.tan(x*(math.pi/2-1e-4))
        x = tf.nn.softmax(inputs+x, axis=self.axis)

        return x
    
class SqueezeDimLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)

        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)
    
class QuerySamplesFeaturesMHAUpdate(tf.keras.layers.Layer):
    def __init__(self, binarised=False, bin_reg=1e2, return_weights=False, **kwargs):
        super().__init__(**kwargs)

        self.binarised = binarised
        self.return_weights = return_weights

        if binarised:
            self.binarize = BinarisedSoftmax(axis=-1, reg=bin_reg)

    def build(self, input_shape):

        samples_input_shape, scores_input_shape = input_shape[0], input_shape[1]

        num_heads = scores_input_shape[-3]
        embs_dim = samples_input_shape[-1]

        self.V_d = tf.keras.layers.Dense(embs_dim)
        self.O_d = tf.keras.layers.Dense(embs_dim)

        self.V_head_extractior = HeadsPermuter(num_heads, reverse=False)
        self.output_perm = HeadsPermuter(num_heads, reverse=True)

    def call(self, inputs, pos_enc=None):

        sample_features, scores = inputs[0], inputs[1]

        V = self.V_head_extractior(self.V_d(tf.transpose(sample_features, perm=[0,2,1,3])))
        if pos_enc is not None:
            V += tf.expand_dims(tf.transpose(pos_enc, perm=[0,2,1,3]), axis=-3)

        scores = tf.transpose(scores, perm=[0,4,2,3,1])
        if self.binarised:
            weights = self.binarize(scores)
        else:
            weights = tf.nn.softmax(scores,axis=-1)

        V = tf.matmul(weights, V)

        out = self.O_d(self.output_perm(V))
        out = tf.transpose(out, perm=[0,2,1,3])
        if self.return_weights:
            return out, weights
        return out

def calc_2x2_vec_angle(x):
    return tf.squeeze(tf.math.atan2(*tf.split(tf.squeeze(tf.subtract(*tf.split(x, 2, axis=-2)), axis=-2), 2, axis=-1)), axis=-1)


class AngleHeadedSineEncoding(tf.keras.layers.Layer):
    def __init__(self, emb_size, size, temperature=1e2, **kwargs):
        super().__init__(**kwargs)

        self.size = size
        self.emb_size = emb_size
        self.temperature = temperature
        num_pos_features = emb_size//2

        self.yx = tf.transpose(tf.reshape(xy_coords((size, size))[...,::-1], (size**2, 2)), perm=[1,0])[tf.newaxis, tf.newaxis]-(size-1)/2

        dim_t = tf.math.cumsum(tf.ones((num_pos_features,)))-1
        self.dim_t = temperature ** (2 * (dim_t // 2) / num_pos_features)

    def build(self, input_shape):

        self.angles_num = input_shape[-1]

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        rot_matrix = tf.reshape(tf.stack([tf.cos(inputs), tf.sin(inputs),-tf.sin(inputs), tf.cos(inputs)], axis=-1), (B, self.angles_num, 2, 2))

        yx_rot = tf.transpose(tf.matmul(rot_matrix, self.yx), [0,1,3,2])
        
        pos_yx = yx_rot[...,tf.newaxis]/self.dim_t

        pos_emb = tf.reshape(tf.concat([
                tf.sin(pos_yx[...,0::2]),
                tf.cos(pos_yx[...,1::2])
            ], axis=-1), (B,self.angles_num, self.size**2, self.emb_size))
        
        return pos_emb