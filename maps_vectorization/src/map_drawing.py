import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from google.cloud import bigquery
from google.auth import default
import pyproj
import shapely
from shapely import ops
import numpy as np
import math
import random
from PIL import Image, ImageFont, ImageDraw
from patterns import text_label_randomization, gen_colors, drawing_patterns
import copy
import cv2 as cv
import warnings
from scipy.stats import skewnorm


####

######### PARCEL INPUT GENERATOR ###########

####

### functions for parcel size adjustment

def bbox_size_from_input_shape(shape):
    # shape input [N,1,2]
    XY1 = np.min(shape, axis=0)[0]
    XY2 = np.max(shape, axis=0)[0]
    sizes = XY2-XY1
    return np.mean(sizes)

def relative_input_bbox_sizes(shapes, img_size):
    img_size = np.mean(img_size)

    return np.array([bbox_size_from_input_shape(shape) for shape in shapes])/img_size

def bbox_center(shape):
    # input shape [N,1,2]
    # output shape [2]
    XY1 = np.min(shape, axis=0)[0]
    XY2 = np.max(shape, axis=0)[0]
    return XY1+(XY2-XY1)/2

def bbox_from_input_shape(shape):
    # input shape [N,1,2]
    # output shape [Xmin,Ymin,Xmax,Ymax]
    XY1 = np.min(shape, axis=0)[0]
    XY2 = np.max(shape, axis=0)[0]
    return np.concatenate([XY1, XY2], axis=0)

def crop_shape_by_window(shape, window_polygon):
    shape_polygon = shapely.Polygon(shape[:,0])
    if shapely.intersects(shape_polygon, window_polygon):
        try:
            shape_polygon = shapely.intersection(shape_polygon, window_polygon)
            coords = np.array([[x,y] for x,y in zip(*shape_polygon.exterior.xy)])[:,np.newaxis]
            if len(coords)>2:
                return coords
            else:
                return None
        except:
            return None
    else:
        return None

def cropping_shape_input(shapes, img_size, padding, target_mean_size, min_shapes_num, adjust_extending=False):
    bbox_sizes = relative_input_bbox_sizes(shapes, img_size)
    centroids = np.stack([bbox_center(shape) for shape in shapes], axis=0)
    bboxes = np.stack([bbox_from_input_shape(shape) for shape in shapes], axis=0)
    init_window_size = img_size-2*padding

    filtered_shapes = shapes.copy()
    approx_mean_size = np.mean(bbox_sizes)
    shapes_num = len(shapes)
    scaler = 1.0
    
    while (shapes_num>min_shapes_num) & (approx_mean_size<target_mean_size):
        center = np.mean(centroids, axis=0, keepdims=True)
        corners = [bboxes[:,[x,y]] for x in [0,2] for y in [1,3]]
        corner_scores = [np.mean(np.abs(corner-center), axis=-1) for corner in corners]
        corner_scores_agg = [np.max(s) for s in corner_scores]
        delete_idx = np.argmax(corner_scores[np.argmax(corner_scores_agg)])

        bbox_sizes = np.delete(bbox_sizes, delete_idx, axis=0)
        bboxes = np.delete(bboxes, delete_idx, axis=0)
        centroids = np.delete(centroids, delete_idx, axis=0)
        filtered_shapes.pop(delete_idx)
        
        new_window_size = np.max(bboxes[:,2:], axis=0)-np.min(bboxes[:,:2], axis=0)
        scaler = init_window_size/new_window_size
        approx_mean_size = np.mean(bbox_sizes)*np.mean(scaler)
        shapes_num -= 1

    left_upper = np.min(bboxes[:,:2], axis=0)

    if adjust_extending:
        right_bottom = np.max(bboxes[:,2:], axis=0)
        window = np.concatenate([left_upper, right_bottom], axis=0)
        window_polygon = shapely.Polygon([window[[x,y]] for x,y in zip([0,0,2,2],[1,3,3,1])])
        filtered_shapes = [shape for shape in [crop_shape_by_window(shape, window_polygon) for shape in shapes] if shape is not None]

    left_upper = left_upper[np.newaxis, np.newaxis]
    rescaled_shapes = [((shape-left_upper)*scaler).astype(np.int32)+padding for shape in filtered_shapes]

    return rescaled_shapes

### input generator class

class map_drawer_input_generator:
    def __init__(self, client, randomize_args, batch_size, test_mode, adjustment, adjustment_args):
        self.client = client
        self.randomize_args = randomize_args
        self.batch_size = batch_size
        self.test_mode = test_mode

        self.adjustment = adjustment
        self.adjustment_args = adjustment_args # target_mean_size:float, min_shapes_num:int, adjust_extending:bool

        self.query = '''
                with a as (
                    SELECT ST_CENTROID(geom) cent
                    FROM geo.parcels
                    ORDER BY RAND()
                    LIMIT {batch_size}
                    ),
                    b as (
                    SELECT *, ROW_NUMBER() OVER() group_id
                    FROM a
                    )
                SELECT b.group_id, p.parcel_id, p.geom
                FROM b
                JOIN geo.parcels p ON ST_DWITHIN(b.cent, p.geom, {radius})
            '''
        
        # transform from CRS 4326 to 2180
        self.project_to_pl = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), pyproj.CRS('EPSG:2180'), always_xy=True).transform

        # prepare first batch input
        self.download_batch()

    @staticmethod
    def _get_randomized_params(batch_size, radius_range, target_map_size_range, padding_range):
        radius = np.random.randint(*radius_range)
        target_map_sizes = np.random.randint(*target_map_size_range, batch_size)
        paddings = np.random.randint(*padding_range, batch_size)

        return radius, target_map_sizes, paddings

    def _convert_geom(self, wkt):
        shape = ops.transform(self.project_to_pl, wkt)
        return np.array([[x,y] for x,y in zip(*shape.exterior.xy)])
    
    @staticmethod
    def _prepare_example(shapes, target_map_size, padding):

        corner = np.abs(np.min([np.min(x*np.array([[1,-1]]), axis=0) for x in shapes], axis=0)) #-np.array([[padding,-padding]])
        shapes = [np.abs(shape-corner).reshape((-1,1,2)).astype(np.int32) for shape in shapes]
        
        img_size = np.max([np.max(shape, axis=0)[0] for shape in shapes], axis=0)

        # scale map
        scaling_param = (target_map_size-2*padding)/max(img_size)

        shapes = [np.round(shape*scaling_param).astype(np.int32)+padding for shape in shapes]
        img_size = np.max([np.max(shape, axis=0)[0] for shape in shapes], axis=0)+padding

        return {
            'shapes': shapes,
            'img_size': img_size,
            'padding': padding
            }

    def download_batch(self, ):
        # randomize map parameters
        radius, target_map_sizes, paddings = self._get_randomized_params(self.batch_size, **self.randomize_args)

        # download sets of parcels
        result = self.client.query(self.query.format(batch_size=self.batch_size, radius=radius))
        parcels = result.to_dataframe(geography_as_object=True)

        # convert and group geoms
        print('Parcels memory usage: %.3f MB' % (parcels.memory_usage(deep=True).sum()*1e-6))
        parcels['geom'] = parcels.apply(lambda x: self._convert_geom(x['geom']), axis=1)

        parcels = parcels.groupby('group_id')['geom'].apply(list).to_list()

        # prepare examples collections
        self.parcels = [self._prepare_example(*example_set) for example_set in zip(parcels, target_map_sizes, paddings)]
        if self.adjustment:
            self.backgrounds = self.parcels.copy()
            self.parcels = [{'shapes': cropping_shape_input(**example_set, **self.adjustment_args), 'img_size': example_set['img_size'], 'padding': example_set['padding']} 
                            for example_set in self.parcels]
            self.batch_iter = iter(zip(self.parcels, self.backgrounds[::-1]))
        else:
            self.batch_iter = iter(zip(self.parcels, self.parcels[::-1]))

    def __next__(self):
        try:
            return next(self.batch_iter)
        except:
            if self.test_mode:
                random.shuffle(self.parcels)
                if self.adjustment:
                    random.shuffle(self.backgrounds)
                    self.batch_iter = iter(zip(self.parcels, self.backgrounds))
                else:
                    self.batch_iter = iter(zip(self.parcels, self.parcels[::-1]))
            else:
                self.download_batch()
            return next(self.batch_iter)



####

######### POSITION MAP PARAMETERS RANDOMIZATION ###########

####


class random_type:
    def __init__(self, types_prob_map):

        self.types, self.probs = self._map_prob_keys(types_prob_map)

    @staticmethod
    def _map_prob_keys(mapping):
        return np.transpose(np.array([[key, mapping[key]] for key in mapping.keys()], dtype='object'), axes=[1,0])
    
    def random_types(self, size):
        return np.random.choice(self.types, size=size, p=self.probs.astype(np.float32))
        

class map_drawer_arg_randomizer:
    def __init__(self, 
                 line_border_types_probs,
                 line_filled_types_probs,
                 single_shape_prob,
                 multishape_range):
        
        self.line_border_types_gen = random_type(line_border_types_probs)
        self.line_filled_types_gen = random_type(line_filled_types_probs)
        self.single_shape_prob = single_shape_prob
        self.multishape_range = multishape_range

    @staticmethod
    def _split_types_to_columns(info):
        pattern_type = info['pattern_style']['pattern_type']
        filled = info if pattern_type in ['solid', 'striped'] else None
        line_filled = info if pattern_type=='line_filled' else None
        line_border = info if pattern_type=='line_border' else None

        return [filled, line_filled, line_border]
    
    def _split_types_to_lists(self, patterns_info):
        return [[x for x in arr if x!=None] for arr in np.transpose(np.array([self._split_types_to_columns(r) for r in patterns_info], dtype='object'), axes=[1,0])]
    
    def _gen_shapes_num(self,):
        return np.random.randint(*self.multishape_range) if np.random.binomial(1, self.single_shape_prob) else 1 
        
    def _prepare_drawing_parameters(self, patterns_info):
        for r in patterns_info: r['map_args']={}
        filled, line_filled, line_border = self._split_types_to_lists(patterns_info)

        for r in filled: r['map_args']['shape_type']='parcel_polygon'

        for r, shape_type in zip(line_filled, self.line_filled_types_gen.random_types(len(line_filled))): r['map_args']['shape_type']=shape_type

        for r, shape_type in zip(line_border, self.line_border_types_gen.random_types(len(line_border))): r['map_args']['shape_type']=shape_type

        updated_info = filled + line_filled + line_border

        for r in updated_info: 
            r['map_args']['shapes_num']=self._gen_shapes_num() if r['map_args']['shape_type']!='random_polygon' else 1
            r['map_args']['transparent_paste'] = r['pattern_style']['pattern_type'] in ('line_border', 'line_filled')

        return updated_info




####

######### RANDOM SHAPES ###########

####




class shapes_generator:
    def __init__(self, img_size, padding, radius_range, quarter_vertices_range, max_dim_coverage_ratio, dim_diff_range, edges_range, min_distance):
        self.width, self.height = img_size
        self.padding = padding
        self.radius_range = radius_range
        self.quarter_vertices_range = quarter_vertices_range
        self.max_dim_coverage_ratio = max_dim_coverage_ratio

        self.dim_diff_range = dim_diff_range
        self.edges_range = edges_range
        self.min_distance = min_distance


    def random_polygon(self, ):
        angles = np.concatenate([(np.sort(np.random.uniform(0,0.5, size=np.random.randint(*self.quarter_vertices_range)))+q)*math.pi for q in [-1,-0.5,0,0.5]], axis=0)
        
        radiuses = np.random.uniform(*self.radius_range, size=len(angles))

        x = np.round(radiuses*np.sin(angles),0).astype(np.int32)[:,np.newaxis]
        y = np.round(radiuses*np.cos(angles),0).astype(np.int32)[:,np.newaxis]
        pts = np.concatenate([x,y], axis=-1)
        pts = np.append(pts, pts[0,np.newaxis], axis=0)

        max_size = (np.array([self.width, self.height])-2*self.padding)*self.max_dim_coverage_ratio-1
        polygon_size = np.max(pts, 0)-np.min(pts,0)
        rescale = max_size/polygon_size
        rescale[rescale>1.0] = 1.0

        pts = pts*rescale

        polygon_size = np.max(pts, 0)-np.min(pts,0)
        corner_x = np.random.randint(self.padding, self.width-self.padding-polygon_size[0])
        corner_y = np.random.randint(self.padding, self.height-self.padding-polygon_size[1])
        corner = np.array([[corner_x, corner_y]], np.int32)

        pts = np.append(pts, pts[0,np.newaxis], axis=0)

        return (pts-np.min(pts,0)[np.newaxis]+corner).reshape((-1,1,2)).astype(np.int32)
    
    def random_line(self, ):
        max_x = self.width-2*self.padding
        max_y = self.height-2*self.padding

        line_points = []
        edges_num = np.random.randint(*self.edges_range)
        starting_point = [np.random.randint(0,max_x), np.random.randint(0, max_y)]
        line_points.append(starting_point)

        for _ in range(edges_num):
            i = 0
            while True:
                new_x = max(0, min(max_x, starting_point[0] + np.random.randint(*self.dim_diff_range)*(np.random.binomial(1,0.5)*2-1)))
                new_y = max(0, min(max_y, starting_point[1] + np.random.randint(*self.dim_diff_range)*(np.random.binomial(1,0.5)*2-1)))
                new_point = [new_x, new_y]

                if len(line_points)>2:
                    curr_linestring = shapely.LineString(line_points[:-1])
                    new_line = shapely.LineString([starting_point, new_point])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if shapely.distance(curr_linestring, new_line)>self.min_distance:
                            line_points.append(new_point)
                            starting_point = new_point
                            break
                else:
                    line_points.append(new_point)
                    starting_point = new_point
                    break

                if i>5:
                    break
                i += 1

        return (np.array(line_points)+self.padding).reshape((-1,1,2))
    


####

######### MAP DRAWER CLASS ###########

####



class map_drawer:
    def __init__(self, args_randomizer, random_shape_drawer_args, padding, shapes, img_size, patterns_info, background, map_drawing_args, map_background_label_randomize_args):
        
        self.patterns_info = args_randomizer._prepare_drawing_parameters(patterns_info)

        self.img_size = img_size

        self.background = background

        self.background_color = np.random.randint(245,256)
        self.img = Image.fromarray(np.ones(img_size.tolist()[::-1]+[3], dtype=np.uint8)*self.background_color)
        self.draw = ImageDraw.Draw(self.img)

        self.shape_generator = shapes_generator(img_size, padding, **random_shape_drawer_args)

        self.background_label_generator = text_label_randomization(map_background_label_randomize_args, img_size.tolist())

        self.map_drawing_args = map_drawing_args

        self.shapes, self.probs = self._compute_probs(shapes)

    def _compute_probs(self, shapes):
        shapes_num = len(shapes)
        bbox_sizes = relative_input_bbox_sizes(shapes, self.img_size)

        ## filter out small shapes
        max_shapes_to_delete = max(0, shapes_num-self.map_drawing_args['min_shapes_num'])

        small_bbox_idxs = np.where(bbox_sizes<self.map_drawing_args['minimal_bbox_size'])[0]
        if (max_shapes_to_delete>0) & (len(small_bbox_idxs)>0):
            idxs_to_delete = np.random.choice(small_bbox_idxs, size=min(max_shapes_to_delete, len(small_bbox_idxs)), replace=False)
            bbox_sizes = np.delete(bbox_sizes, idxs_to_delete, axis=0)
            shapes_num = len(bbox_sizes)
            shapes = [shape for i, shape in enumerate(shapes) if i not in idxs_to_delete]
        #######

        probs = skewnorm.pdf(bbox_sizes, **self.map_drawing_args['distr_args'])+1e-5
        probs = probs/np.sum(probs)

        return shapes, probs

    def _assign_parcels(self, ):
        all_shapes_num = len(self.shapes)
        available_shape_idxs = list(range(all_shapes_num))
        available_shape_probs = self.probs
        all_shapes_idxs = list(range(all_shapes_num))

        for info in self.patterns_info:
            shape_type, shapes_num, transparent = info['map_args'].values()

            if shape_type=='parcel_polygon':
                shapes_num = min(shapes_num, len(available_shape_idxs)) if not transparent else min(shapes_num, all_shapes_num)
                if shapes_num>0:
                    p = self.probs if transparent else available_shape_probs
                    selected_parcels_idxs = np.random.choice(all_shapes_idxs if transparent else available_shape_idxs, size=shapes_num, replace=False, p=p)
                    info['map_args']['shapes'] = copy.deepcopy(list(map(self.shapes.__getitem__, selected_parcels_idxs)))
                    if not transparent:
                        available_shape_idxs = [idx for idx in available_shape_idxs if idx not in selected_parcels_idxs]
                        available_shape_probs = self.probs[available_shape_idxs].copy()
                        available_shape_probs = available_shape_probs/np.sum(available_shape_probs)
                else:
                    info['map_args']['shapes'] = []
            elif shape_type=='random_polygon':
                info['map_args']['shapes'] = [self.shape_generator.random_polygon() for _ in range(shapes_num)]

            elif shape_type=='random_line':
                info['map_args']['shapes'] = [self.shape_generator.random_line() for _ in range(shapes_num)]

    @staticmethod
    def _safe_mean(x):
        if np.all(np.isnan(x)):
            return 0
        else:
            return np.nanmean(x)

    def _random_text_label_position4polygon(self, pts):
        corner = np.min(pts, axis=0)
        x0, y0 = [0,0]
        x1, y1 = np.max(pts, axis=0)[0]-corner[0]

        y_l, y_r = y0 + (np.random.uniform(0, 1, 2)*y1).astype(np.int32)
        cross_line = shapely.LineString([[x0,y_l],[x1,y_r]])
        shape = shapely.Polygon(np.squeeze(pts, axis=1)-corner)

        try:
            cross_line = shapely.intersection(cross_line, shape)
        except:
            cross_line = cross_line.centroid
        geom_num = shapely.get_num_geometries(cross_line)
        if geom_num>1:
            cross_line = shapely.get_geometry(cross_line, np.random.randint(0,geom_num))
        xs, ys = cross_line.xy

        try:
            pos_x = int(self._safe_mean(xs))#int(np.nanmean(xs))
            pos_y = int(self._safe_mean(ys))#int(np.nanmean(ys))
        except:
            pos_x, pos_y = (0,0)

        return (pos_x, pos_y)
    
    def _random_text_label_position4line(self, pts):
        ep_id = np.random.randint(1, len(pts))
        sp, ep = np.squeeze(pts[ep_id-1:ep_id+1]-np.min(pts,0)[np.newaxis], axis=1)
        pos = sp + ((ep-sp)*np.random.uniform(0,1)).astype(np.int32) + np.random.randint(-50,50,2)

        return tuple(pos.tolist())
    
    def _draw_background_parcels(self, color, shapes, img_size, **kwargs):
        padding = ((self.img_size-img_size)/2).astype(np.int32)[np.newaxis, np.newaxis]
        cv.polylines(self.img, [shape+padding for shape in shapes], True, color=color)

    def _gen_background_args(self, draw_parcels_prob, draw_background_prob, grayscale_prob, light_limit, background_labels_prob, background_labels_range, **kwargs):
        draw_parcels = np.random.binomial(1, draw_parcels_prob)
        draw_background = np.random.binomial(1, draw_background_prob)

        parcels_color, background_color = [gen_colors(light_limit, grayscale) for grayscale in np.random.binomial(1, grayscale_prob, 2)]

        background_labels_num = np.random.randint(*background_labels_range) if np.random.binomial(1, background_labels_prob) else 0

        return draw_parcels, draw_background, parcels_color, background_color, background_labels_num

    def _draw_background_labels(self, label_text, font, color, stroke, pos, **kwargs):
        self.draw.text(pos, label_text, fill=color, align='center', anchor='mm', stroke_width=stroke, stroke_fill=(np.random.randint(245,256),)*3,
                    font=font)

    def draw_map(self, ):
        self._assign_parcels()
        self.pattern_drawer = drawing_patterns(self.img)

        for info in self.patterns_info:
            line_border_args = info['line_border_args']
            if type(line_border_args).__name__!='NoneType':
                padding_size = sum(map(line_border_args.get, ['height','lineshape_thickness','solid_line_thickness']))
            else:
                padding_size = 0
            padding = np.array([padding_size]*2)

            if type(info['text_label_style']).__name__!='NoneType':
                info['text_label_style']['legend_pos'] = info['text_label_style']['pos']
                info['text_label_style']['abs_pos_on_map'] = []
            
            for shape in info['map_args']['shapes']:
                pattern_xy = np.min(shape, axis=0)[0]

                if type(info['text_label_style']).__name__!='NoneType':
                    shape_type = info['map_args']['shape_type']
                    if shape_type!='random_line':
                        info['text_label_style']['pos'] = self._random_text_label_position4polygon(shape)
                    else:
                        info['text_label_style']['pos'] = self._random_text_label_position4line(shape)
                    info['text_label_style']['abs_pos_on_map'].append([d+s for d,s in zip(info['text_label_style']['pos'], (pattern_xy-padding))])
                

                self.pattern_drawer.set_pattern_size(*(np.max(shape, axis=0)[0]-pattern_xy+2*padding).tolist())
                
                self.pattern_drawer.draw_single_pattern(shape[::-1]-pattern_xy[np.newaxis,np.newaxis]+padding, 
                                                   info['pattern_style'], 
                                                   tuple((pattern_xy-padding).tolist()), 
                                                   info['fill_args'], 
                                                   info['line_border_args'],
                                                   info['text_label_style'], 
                                                   legend_pattern=False, 
                                                   shape_only_paste=True, 
                                                   transparent_paste= True) #info['map_args']['transparent_paste'])
                
        #background drawing
        draw_parcels, draw_background, parcels_color, background_color, background_labels_num = self._gen_background_args(**self.map_drawing_args)
        label_style = self.background_label_generator.random_text_label_args()
        del label_style['pos']
        for _ in range(background_labels_num):
            pos = tuple([np.random.randint(0, d) for d in self.img_size])
            label_text = self.background_label_generator.random_label_text()
            self._draw_background_labels(label_text=label_text, pos=pos, **label_style)
        self.img = np.array(self.img)
        if draw_parcels:
            self._draw_background_parcels(parcels_color, self.shapes, self.img_size)
        if draw_background:
            self._draw_background_parcels(background_color, **self.background)
