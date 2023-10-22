import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from models_src.Mask_RCNN import NonMaxSupression, gen_anchors

class F12D(tf.keras.metrics.Metric):
    def __init__(self, threshold, name='F1', **kwargs):
        super(F12D, self).__init__(name=name, **kwargs)

        self.f1 = tf.keras.metrics.F1Score(threshold=threshold, average='micro')
        self.flatten = tf.keras.layers.Flatten()
        self.score = self.add_weight(name='f1', initializer='zeros')
        self.iterations = self.add_weight(name='iters', initializer='zeros')
        self.threshold = threshold

    def get_config(self):
        return {**super().get_config(), 'threshold': self.threshold}

    def update_state(self, y_true, y_pred, sample_weight=None):

        self.score.assign_add(self.f1(self.flatten(tf.cast(y_true, tf.float32)), self.flatten(y_pred)))
        self.iterations.assign_add(1.0)

    def result(self):
        return self.score/self.iterations
    
class WeightedF12D(tf.keras.metrics.Metric):
    def __init__(self, name='F1', threshold=0.5, average='micro'):
        super().__init__(name=name)

        self.score = tf.keras.metrics.Mean()
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        prec = self.precision(y_true, y_pred, sample_weight)
        rec = self.recall(y_true, y_pred, sample_weight)

        score = tf.math.divide_no_nan(2*prec*rec, prec+rec)

        self.score.update_state(score)

    def reset_state(self):
        self.score.reset_state()
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        return self.score.result()

def extract_coords(bbox):
        return bbox[...,0], bbox[...,1], bbox[...,2], bbox[...,3]
    
def calc_area(Y1,X1,Y2,X2):
        return (X2-X1)*(Y2-Y1)

def IoU(a,b):
    aY1, aX1, aY2, aX2 = extract_coords(a)
    bY1, bX1, bY2, bX2 = extract_coords(b)

    xI = tf.clip_by_value(bX2, aX1, aX2) - tf.clip_by_value(bX1, aX1, aX2)
    yI = tf.clip_by_value(bY2, aY1, aY2) - tf.clip_by_value(bY1, aY1, aY2)

    I = xI*yI
    U = calc_area(aY1, aX1, aY2, aX2)+calc_area(bY1, bX1, bY2, bX2)-I

    return tf.math.divide_no_nan(I,U)
    

class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, name='IoU'):
        super().__init__(name=name)

        self.score = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):

        scores = tf.expand_dims(IoU(y_true, y_pred), axis=-1)

        self.score.update_state(scores, sample_weight=sample_weight)

    def reset_state(self):
        self.score.reset_state()

    def result(self):
        return self.score.result()


class HungarianMatching:
    def __init__(self, matching_loss_func=tf.keras.losses.BinaryCrossentropy(), pool_size=4, perm=[0,2,1]):

        self.matching_loss = matching_loss_func
        self.matching_loss.reduction = tf.keras.losses.Reduction.NONE

        self.pool_size = pool_size
        self.perm = perm

    def _calc_losses(self, cost_matrix):
        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(linear_sum_assignment(x)), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int32)
        return tf.gather_nd(cost_matrix, match_idxs, batch_dims=1)
    
    def _gen_matching(self, cost_matrix, y_true, y_pred):
        match_idxs = tf.map_fn(lambda x: tf.convert_to_tensor(linear_sum_assignment(x)), elems=cost_matrix, fn_output_signature=tf.int32)
        return tf.gather(tf.squeeze(y_true, axis=1), match_idxs[:,1], batch_dims=1, axis=1), \
            tf.gather(tf.squeeze(y_pred, axis=2), match_idxs[:,0], batch_dims=1, axis=1)

    @tf.function
    def __call__(self, y_true, y_pred):
        if self.pool_size>1:
            y_true_pooled = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.pool_size)(y_true)
            y_pred_pooled = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.pool_size)(y_pred)
        else:
            y_true_pooled = y_true
            y_pred_pooled = y_pred

        pooled_shape = tf.shape(y_true_pooled)
        B, T = pooled_shape[0], pooled_shape[-1]
        P = tf.shape(y_pred_pooled)[-1]

        y_true_pooled = tf.expand_dims(tf.transpose(tf.reshape(y_true_pooled, (B,-1,T)), perm=self.perm), axis=1)
        y_pred_pooled = tf.expand_dims(tf.transpose(tf.reshape(y_pred_pooled, (B,-1,P)), perm=self.perm), axis=2)

        cost_matrix = self.matching_loss(y_true_pooled, y_pred_pooled)

        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(tf.numpy_function(linear_sum_assignment, [x], [tf.int64, tf.int64])), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int64)

        return match_idxs


class HungarianLoss(tf.keras.losses.Loss):
    def __init__(self, hungarian_matching, loss_func=tf.keras.losses.BinaryCrossentropy(), reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        super().__init__(name="HL",reduction=reduction,**kwargs)
        
        self.hungarian_matching = hungarian_matching
        self.loss_func = loss_func

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        match_idxs = tf.stop_gradient(self.hungarian_matching(y_true, y_pred))
        y_true = tf.gather(y_true, match_idxs[:,:,1], batch_dims=1, axis=-1)
        y_pred = tf.gather(y_pred, match_idxs[:,:,0], batch_dims=1, axis=-1)
        return self.loss_func(y_true, y_pred)
    
class HungarianClassificationMetric(tf.keras.metrics.Metric):
    def __init__(self, hungarian_matching, metric_func, name='ClassMetric', **kwargs):
        super(HungarianClassificationMetric, self).__init__(name=name, **kwargs)

        self.hungarian_matching = hungarian_matching
        self.metric_func = metric_func
        #self.flatten = tf.keras.layers.Flatten()

        self.score = self.add_weight(name=name+'_score', initializer='zeros') #tf.Variable(0, dtype=tf.float32)#
        self.iterations = self.add_weight(name=name+'_iters', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        shape = tf.shape(y_true)
        B, H, W, T = shape[0], shape[1], shape[2], shape[3]

        match_idxs = tf.stop_gradient(self.hungarian_matching(y_true, y_pred))
        y_true = tf.reshape(tf.gather(y_true, match_idxs[:,:,1], batch_dims=1, axis=-1), (B, H*W*T))
        y_pred = tf.reshape(tf.gather(y_pred, match_idxs[:,:,0], batch_dims=1, axis=-1), (B, H*W*T))
        self.score.assign_add(self.metric_func(y_true, y_pred))
        self.iterations.assign_add(1.0)

    def reset_states(self):
        self.score.assign(0.0)
        self.iterations.assign(0.0)

    def result(self):
        return self.score/self.iterations
    

class MultivariantHungarianLoss():
    def __init__(self,
                 classification=True,
                 bbox=True,
                 mask=True,
                 losses_weights=[1,1,1],
                 output_proposals=6,
                 output_mask_size=(256,256),
                 iou_weight=0.5,
                 bbox_regularization_rank=1,
                 mask_alpha=0.75,
                 mask_gamma=5,
                 mask_pool_size=8,
                 mask_class_pred=False,
                 return_matching=True,
                 name='HL',
                 class_smoothing=0.0,
                 mask_smoothing=0.0,
                 unit_class_matrix=False,
                 **kwargs):
        #super(MultivariantHungarianLoss, self).__init__(name='HL', **kwargs)
        self.name = name

        self.iou_weight = iou_weight
        self.bbox_reg_rank = bbox_regularization_rank
        self.unit_class_matrix = unit_class_matrix


        class_loss = tf.keras.losses.BinaryCrossentropy(reduction='none', label_smoothing=class_smoothing)
        mask_loss = tf.keras.losses.BinaryFocalCrossentropy(reduction='none', alpha=mask_alpha, gamma=mask_gamma, label_smoothing=mask_smoothing)
        flags=[classification, bbox, mask]
        self.inputs_num = sum(flags)

        self.cost_matrix_args, self.masking_flags, self.input_names, self.matching_args, self.losses_weights = self._gen_cost_matrix_args(flags=flags,
                                            losses=[class_loss, self.BBoxLoss, mask_loss],
                                            pool_sizes=[1,1,mask_pool_size],
                                            perms=[[0,1],[0,1,2],[0,3,1,2]],
                                            class_masks=[mask_class_pred, True, True],
                                            input_names=['class', 'bbox', 'mask'],
                                            output_sizes=[(output_proposals,1), (output_proposals,4), (output_proposals,output_mask_size[0]*output_mask_size[1])],
                                            losses_weights=losses_weights
        )
        
        self.classification = classification

        self.return_matching = return_matching

        self.pool = tf.keras.layers.AveragePooling2D(pool_size=mask_pool_size, strides=mask_pool_size)

    ##### Cost Matrix #####

    def _gen_cost_matrix_args(self, flags, losses, pool_sizes, perms, class_masks, input_names, output_sizes, losses_weights):
        output_args = []
        masking_flags = []
        names = []
        matching_args = []
        losses_weights_list = []
        for flag, loss, pool_size, perm, class_mask, name, output_size, loss_weight in zip(flags, losses, pool_sizes, perms, class_masks, input_names, output_sizes, losses_weights):
            if flag:
                output_args.append({
                    'cost_func': loss,
                    'pool_size': pool_size,
                    'perm': perm,
                    'unit_true': name=='class' if self.unit_class_matrix else False
                    })
                masking_flags.append(class_mask)
                names.append(name)

                matching_args.append({
                    'class_mask': class_mask,
                    'perm': perm,
                    'output_size': output_size
                })
                losses_weights_list.append(loss_weight)
        
        losses_weights_list = tf.constant(losses_weights_list, dtype=tf.float32)
        losses_weights_list /= tf.reduce_sum(losses_weights_list)

        return output_args, masking_flags, names, matching_args, losses_weights_list
    
    def _calc_cost_matrix(self, a, b, cost_func,pool_size=1, perm=[0,1,2], unit_true=False):

        if pool_size>1:
            a = self.pool(a)
            b = self.pool(b)
            
        a = tf.transpose(a, perm=perm)
        b = tf.transpose(b, perm=perm)
        a_shape = tf.shape(a)
        B, T = a_shape[0], a_shape[1]
        P = tf.shape(b)[1]

        if unit_true:
            a = tf.ones_like(a, dtype=tf.float32)

        a = tf.repeat(tf.expand_dims(tf.reshape(a, (B,T,-1)), axis=1), P, axis=1)
        b = tf.repeat(tf.expand_dims(tf.reshape(b, (B,P,-1)), axis=2), T, axis=2)
        cost_matrix = cost_func(a, b)
        
        return cost_matrix


    ##### BBox Loss #####

    ### IoU
    def IoULoss(self, a, b):
        return 1-tf.clip_by_value(IoU(a,b), 1e-5, 1-1e-5)
    
    ### Bbox regression
    def Ln(self, a, b):
        return tf.reduce_mean(tf.abs((a-b)**self.bbox_reg_rank), axis=-1)
    
    ### Combined BBox Loss
    def BBoxLoss(self, a, b):
        return self.iou_weight*self.IoULoss(a,b) + (1-self.iou_weight)*self.Ln(a,b)
    

    ##### Hungarian Matching #####

    @tf.autograph.experimental.do_not_convert
    def HungarianMatching(self, cost_matrix):
        match_idxs = tf.map_fn(lambda x: tf.transpose(tf.convert_to_tensor(tf.numpy_function(linear_sum_assignment, [x], [tf.int64, tf.int64])), perm=[1,0]), elems=cost_matrix, fn_output_signature=tf.int64)
        y_true_idxs = match_idxs[...,1]
        y_pred_idxs = match_idxs[...,0]
        return y_true_idxs, y_pred_idxs
    
    def add(self, cost_matrices):
        cost_matrix = cost_matrices[0] * self.losses_weights[0]
        for i,cm in enumerate(cost_matrices[1:]):
            cost_matrix += cm * self.losses_weights[i+1]
        return cost_matrix
    
    @staticmethod
    def extract_losses(cost_matrix, y_true_idxs, y_pred_idxs):
        return tf.gather(tf.gather(cost_matrix, y_pred_idxs, batch_dims=1, axis=1), y_true_idxs, batch_dims=2, axis=2)
    

    def extract_features(self, features, idxs, mask, class_mask, perm, output_size):
        idxs_shape = tf.shape(idxs)
        B, T = idxs_shape[0], idxs_shape[1]
        features = tf.transpose(features, perm=perm)
        P = tf.shape(features)[1]
        features = tf.reshape(features, (B,P,-1))
        
        features = tf.gather(features, idxs, batch_dims=1, axis=1)

        if class_mask:
            features *= mask
        
        #features = tf.reshape(features, (B,-1))
        features.set_shape((None,)+output_size)
        return features

    ##########

    def get_config(self):
        return {
            'name': self.name,
            'input_names': self.input_names,
            'losses_weights': self.losses_weights,
            'iou_weight': self.iou_weight,
            'bbox_reg_rank': self.bbox_reg_rank,
            'unit_class_matrix': self.unit_class_matrix,
            'masking_flags': self.masking_flags,
            'return_matching': self.return_matching
        }
    
    def __call__(self, y_true, y_pred):
        y_true = [tf.cast(y_true[input_name], tf.float32) for input_name in self.input_names]
        y_pred = list(y_pred.values())

        cost_matrices = [self._calc_cost_matrix(y_true_n, y_pred_n, **args) for y_true_n, y_pred_n, args in zip(y_true, y_pred, self.cost_matrix_args)]

        HM_cost_matrix = self.add(cost_matrices)
        HM_cost_matrix = tf.where(tf.math.is_nan(HM_cost_matrix), 2.0, HM_cost_matrix)


        idxs = self.HungarianMatching(HM_cost_matrix) #tf.stop_gradient()
        y_true_idxs, y_pred_idxs = idxs[0], idxs[1]

        unit_mask = tf.ones_like(y_true_idxs, dtype=tf.float32)
        objects_mask = tf.gather(y_true[0], y_true_idxs, batch_dims=1, axis=-1) if self.classification else unit_mask
        objects_masks = [objects_mask if masking else unit_mask for masking in self.masking_flags]
        denominators = [tf.reduce_sum(om, axis=-1) for om in objects_masks]

        matched_losses = self.add([tf.reduce_sum(self.extract_losses(cm, y_true_idxs, y_pred_idxs)*om, axis=-1)/d for cm, om, d in zip(cost_matrices, objects_masks, denominators)])

        if self.return_matching:
            objects_mask = tf.expand_dims(objects_mask, axis=-1)
            y_true_output = {name: self.extract_features(features, y_true_idxs, None, False, matching_args['perm'], matching_args['output_size']) for 
                             features, matching_args, name in zip(y_true, self.matching_args, self.input_names)}
            y_pred_output = {name: self.extract_features(features, y_pred_idxs, objects_mask, **matching_args) for 
                             features, matching_args, name in zip(y_pred, self.matching_args, self.input_names)}
            output = (matched_losses, y_true_output, y_pred_output)
        else:
            output = matched_losses
        return output
    

class RPNloss():
    def __init__(self, 
                 confidence_score=True,
                 anchor_args=None,
                 normalize_target_scores=False,
                 iou_weight=0.5,
                 init_top_k_proposals=0.5, 
                 iou_threshold=0.5, 
                 output_proposals=200,
                 return_matching=True,
                 label_proposals=30,
                 matching_iou_weight=0.5,
                 confidence_loss = tf.keras.losses.MeanAbsoluteError(),
                 name='RPNL',
                 **kwargs):

        self.name = name
        self.NMS = NonMaxSupression(init_top_k_proposals, iou_threshold, output_proposals)
        self.return_matching = return_matching
        self.matching = MultivariantHungarianLoss(mask=False, losses_weights=[0.,1.,0.], output_proposals=label_proposals, 
                                                  iou_weight=matching_iou_weight, mask_class_pred=True, unit_class_matrix=True)
        
        self.normalize_target_scores = normalize_target_scores
        self.iou_weight = iou_weight
        self.confidence_score = confidence_score

        if confidence_score:
            self.height, self.width = anchor_args['base_img_size']
            bbox_normalization = tf.constant(anchor_args['base_img_size']*2, dtype=tf.float32)[tf.newaxis,tf.newaxis]
            anchors = tf.concat([gen_anchors((None, size, size), anchor_size, window_size, anchor_args['anchor_scales'], anchor_args['base_img_size']) 
                                      for size, anchor_size, window_size in zip(*list(map(anchor_args.get, ['input_sizes', 'anchor_sizes', 'window_sizes'])))], axis=1)
            self.anchors = self._bbox_decoding(anchors)/bbox_normalization
            self.BC = confidence_loss #tf.keras.losses.BinaryCrossentropy(reduction='none')
            self.BC.reduction = tf.keras.losses.Reduction.NONE

    def get_config(self):
        return {
            'name': self.name,
            'init_top_k_proposals': self.NMS.init_top_k_ratio,
            'iou_threshold': self.NMS.iou_threshold,
            'output_proposals': self.NMS.output_proposals,
            'return_matching': self.return_matching
        }
    
    def _bbox_decoding(self, anchor_bboxes):
        YX, HW = anchor_bboxes[...,:2], anchor_bboxes[...,2:]
        HW *= 0.5

        return tf.concat([tf.clip_by_value(YX-HW, [0,0], [self.height, self.width]), tf.clip_by_value(YX+HW, [0,0], [self.height, self.width])], axis=-1)
    
    @staticmethod
    def _binary_crossentropy(a,b):
        b = tf.clip_by_value(b, 1e-5, 1-1e-5)
        return a*tf.math.log(b) + (1-a)*tf.math.log(1-b)
    
    def Ln(self, a, b):
        return tf.reduce_mean(tf.abs((a-b)), axis=-1)

    def __call__(self, y_true, y_pred):
        target_bboxes, target_class = y_true['bbox'], y_true['class'] # [B,T,4], [B,T]
        confidence = y_pred['class'] # [B,P], [B,P,4]
        bboxes = y_pred['bbox'] if not self.confidence_score else self.anchors

        T, P = tf.shape(target_bboxes)[1], tf.shape(bboxes)[1]
        target_bboxes = tf.repeat(tf.expand_dims(target_bboxes, axis=1), P, axis=1) # [B,P,T,4]
        pred_bboxes = tf.repeat(tf.expand_dims(bboxes, 2), T, axis=2) # [B,P,T,4]

        if self.confidence_score:
            B = tf.shape(target_bboxes)[0]
            pred_bboxes = tf.repeat(pred_bboxes, B, axis=0)

        scores = self.iou_weight*IoU(target_bboxes, pred_bboxes)+(1-self.iou_weight)*(1-self.Ln(target_bboxes, pred_bboxes)) #[B,P,T]
        if self.normalize_target_scores:
            max_scores = tf.reduce_max(scores, axis=1, keepdims=True) + 1e-6 # [B,1,T]
            scores /= max_scores

        mask = tf.expand_dims(target_class, axis=1) # [B,1,T]
        scores *= mask
        scores = tf.reduce_max(scores, axis=2)+1e-3 # [B,P]

        if not self.confidence_score:
            weighted_confidence = confidence/tf.reduce_sum(confidence, axis=-1, keepdims=True)#tf.nn.softmax(confidence, axis=-1) # [B,P]
            loss_value = 1-tf.reduce_sum(scores*weighted_confidence, axis=-1)
        else:
            confidence = tf.clip_by_value(confidence, 1e-5, 1-1e-5)
            loss_value = self.BC(scores, confidence)

        if self.return_matching:
            if self.confidence_score:
                bboxes = tf.repeat(bboxes, B, axis=0)
            confidence, bboxes = self.NMS([confidence, bboxes])
            _, y_true_match, y_pred_match = self.matching(y_true,{'class': confidence, 'bbox': bboxes})
            return loss_value, y_true_match, y_pred_match
        return loss_value