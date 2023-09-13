import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from map_drawing import map_drawer_input_generator, map_drawer_arg_randomizer, map_drawer
from legend import load_vocab, gen_random_legend_properties, draw_legend

import numpy as np
import math
import shapely
from PIL import Image, ImageFont, ImageDraw
import json
import tensorflow as tf
import cv2 as cv


####

######### CLOCKWISE COORDS ###########

####



class clockwise_points:
    @staticmethod
    def _north_point(pts):
        north_idxs = np.where(pts[:,1]==np.min(pts[:,1],0))[0]

        north_coords = pts[north_idxs]

        western_idx = np.where(north_coords[:,0]==np.min(north_coords[:,0],0))[0][0]

        startpoint = north_idxs[western_idx]

        return startpoint, pts[startpoint]

    @staticmethod
    def _neighbour_points(pts, idx):
        previous_pos = idx-1 if idx>0 else len(pts)-1
        next_pos = idx+1 if idx<len(pts)-1 else 0

        return pts[[previous_pos, next_pos]]
    
    @staticmethod
    def _calc_angles(x,y):
        angles = np.arctan(np.divide(y,x, out=np.zeros_like(x), where=x!=0))*180/math.pi
        return angles+np.maximum(0,-angles).astype(np.bool8)*180
    
    @staticmethod
    def _linepoints_sorting(pts):
        first, last = pts[[0,-1],0,1]
        direction = 1 if first<last else -1
        return pts[::direction]
    
    def sort_points(self, pts):
        if np.all(pts[0]==pts[-1]):
            pts = np.squeeze(pts[:-1], axis=1)

            npt_idx, npt = self._north_point(pts)

            previous_pos, next_pos = self._neighbour_points(pts, npt_idx)

            prev_angle, next_angle = self._calc_angles(*np.transpose(np.stack([previous_pos, next_pos], axis=0)-npt, axes=[1,0]).astype(np.float32))

            direction = 1 if next_angle<prev_angle else -1

            shifted_pts = np.roll(pts, -npt_idx, axis=0)
            shifted_pts = np.append(shifted_pts, shifted_pts[0,np.newaxis], axis=0)[::direction][:,np.newaxis]
        else:
            shifted_pts = self._linepoints_sorting(pts)

        return shifted_pts




####

######### MAP CONCATENATION ###########

####


class map_concatenation:
    def __init__(self, map_concatenation_args):
        self.map_concatenation_args = map_concatenation_args

        self.target_size = map_concatenation_args['target_size']

        self.clockwise_points = clockwise_points()

    @staticmethod
    def _randomize_concatenation_args(resize_legend_prob, legend_resize_range, target_size, simple_concatenation_prob, horizontal_left_pos_prob, vertical_top_pos_prob, 
                                      minimap_rel_size_range, minimap_min_padding_range, paste_transparent_prob):
        
        legend_resize_ratio = np.random.uniform(*legend_resize_range) if np.random.binomial(1, resize_legend_prob) else 1.0
        minimap_legend_resize_ratio = np.random.uniform(*legend_resize_range) if np.random.binomial(1, resize_legend_prob) else 1.0

        horizontal_left_pos, minimap_horizontal_left_pos = np.random.binomial(1, horizontal_left_pos_prob, 2)
        vertical_top_pos, minimap_vertical_top_pos = np.random.binomial(1, vertical_top_pos_prob, 2)

        minimap_max_dim_size = int(np.random.uniform(*minimap_rel_size_range)*target_size)
        minimap_min_padding = np.random.uniform(*minimap_min_padding_range)

        simple_concatenation = np.random.binomial(1, simple_concatenation_prob)

        paste_transparent = np.random.binomial(1, paste_transparent_prob)

        return {'legend_resize_ratio': legend_resize_ratio,
            'minimap_legend_resize_ratio': minimap_legend_resize_ratio, 
            'horizontal_left_pos': horizontal_left_pos, 
            'minimap_horizontal_left_pos': minimap_horizontal_left_pos, 
            'vertical_top_pos': vertical_top_pos, 
            'minimap_vertical_top_pos': minimap_vertical_top_pos,
            'minimap_max_dim_size': minimap_max_dim_size,
            'minimap_min_padding': minimap_min_padding,
            'simple_concatenation': simple_concatenation,
            'paste_transparent': paste_transparent
            }

    @staticmethod
    def _random_point_outside_shape(shape, img_size, legend_size, padding, forced_vertical_top, forced_horizontal_left):
        
        x0, y0 = [-padding,-padding]
        x1, y1 = [d+padding for d in img_size]

        y_l, y_r = np.random.randint(y0, y1, 2)
        cross_line = shapely.LineString([[x0,y_l],[x1,y_r]])

        cross_line = shapely.difference(cross_line, shape)
        if not shapely.is_empty(cross_line):
            geom_num = shapely.get_num_geometries(cross_line)
            if geom_num>1:
                cross_line = shapely.get_geometry(cross_line, np.random.randint(0,geom_num))
            xs, ys = cross_line.xy

            xs = [int(x) for x in xs]
            xs.sort()
            x0, x1 = xs
            
            ys = [int(y) for y in ys]
            ys.sort()
            y0, y1 = ys

            place = np.random.uniform(0,1)
            x = x0 + int((x1-x0)*place)
            y = y0 + int((y1-y0)*place)
        else:
            x = img_size[0]+int(legend_size[0]/2) if not forced_horizontal_left else -legend_size[0]
            y = img_size[1]+int(legend_size[1]/2) if not forced_vertical_top else -legend_size[1]

        return (x,y)
    
    @staticmethod
    def _calc_buffer_for_img(img_shape):
        return int(round(sum([(d/2)**2 for d in img_shape])**0.5,0))
    
    @staticmethod
    def _white_background_mask(img, pattern_color=(200,200,200)):
        pattern_col_sum = np.abs(np.sum(pattern_color)-255*3)
        mask = (np.abs(np.sum(img.astype(np.int32), axis=-1)-255*3)/pattern_col_sum)
        mask[mask>1] = 1
        return (mask*255).astype(np.uint8)
    
    @staticmethod
    def _mixed_buffer(geom, buffer_arr):
        shift = np.array(buffer_arr)
        shifts = [shift*np.array([x*2-1,y*2-1]) for x in range(2) for y in range(2)]
        points = shapely.get_coordinates(geom)

        return shapely.GeometryCollection([shapely.Point(point+s) for point in points.astype(np.int32) for s in shifts]).convex_hull

    def _random_legend_placement(self, patterns_info, padding, map_size, legend_size, forced_vertical_top, forced_horizontal_left, geoms=None, forced=False):

        # group all drawed shapes into shapely geometry collection
        geom_collection_list = []
        for row_info in patterns_info:
            shape_type = row_info['map_args']['shape_type']
            for shape in row_info['map_args']['shapes']:
                if shape_type!='random_line':
                    geom = shapely.Polygon(np.squeeze(shape, axis=1))
                else:
                    geom = shapely.LineString(np.squeeze(shape, axis=1))
                geom_collection_list.append(geom)
        geom_collection = shapely.geometrycollections(geom_collection_list).convex_hull
        buffer_arr = [int(d/2) for d in legend_size]
        # convex hull 
        ch = self._mixed_buffer(geom_collection, buffer_arr)

        if geoms!=None:
            ch = shapely.union(ch, self._mixed_buffer(geoms, buffer_arr))

        if not forced:
            legend_pos = self._random_point_outside_shape(ch, map_size, legend_size, padding, forced_vertical_top, forced_horizontal_left)
        else:
            x = map_size[0]+int(legend_size[0]/2) if not forced_horizontal_left else -legend_size[0]
            y = map_size[1]+int(legend_size[1]/2) if not forced_vertical_top else -legend_size[1]
            legend_pos = (x,y)
        #print(f'original pos {legend_pos}')
        legend_pos = [center-int(shift/2) for center, shift in zip(legend_pos, legend_size)]

        coords_shift = np.array([0 if d>0 else -d for d in legend_pos])

        top_left_pad = coords_shift[::-1]
        bot_right_pad = np.array([0 if d+l<=m else d+l-m for d,l,m in zip(legend_pos, legend_size, map_size)])[::-1]

        padding4legend = np.append(np.transpose(np.stack([top_left_pad, bot_right_pad]), axes=[1,0]), np.array([[0,0]]), axis=0)

        legend_pos = tuple([0 if d<0 else d for d in legend_pos])

        # legend geometry
        x1,y1 = legend_size
        x0,y0 = legend_pos
        legend_polygon = shapely.Polygon([[x0+x,y0+y] for x,y in [[0,0],[x1,0],[x1,y1],[0,y1]]])

        
        # update shapes coordinates
        coords_shift = coords_shift[np.newaxis, np.newaxis]
        if sum(np.squeeze(coords_shift))!=0:
            for row_info in patterns_info:
                shape_type = row_info['map_args']['shape_type']
                for shape in row_info['map_args']['shapes']:
                    shape += coords_shift


        return legend_pos, padding4legend, coords_shift, legend_polygon
    
    def _rescale_positions(self, patterns_info, legend_shift, map_shift, scale, pad_shift):
        for info in patterns_info:
            for key in ['pattern_xy', 'text_xy']:
                info[key] = [int(round((d+s)*scale,0)) for d,s in zip(info[key], legend_shift)]

            info['textbox_size'] = [int(round(d*scale,0)) for d in info['textbox_size']]
            info['map_args']['shapes'] = [(shape*scale).astype(np.int32) for shape in info['map_args']['shapes']]

            if info['text_label_style']!=None:
                info['text_label_style']['abs_pos_on_map'] = [[int(round((d+s)*scale,0))+p for d,s,p in zip(map_pos, map_shift, pad_shift)] 
                                                              for map_pos in info['text_label_style']['abs_pos_on_map']]

        return patterns_info

    @staticmethod
    def _resize_img(img, max_dim_size):
        original_size = np.array(img.size)
        max_dim = np.max(original_size)
        rescale_ratio = max_dim_size/max_dim
        output_size = np.floor((original_size*rescale_ratio)).astype(np.int32)
        return img.resize(output_size), rescale_ratio
    
    def _calc_img_padding(self, img):
        original_size = np.array(img.size)
        dims_padding = (self.target_size - original_size)[::-1]

        up_left_pad = [int(d/2) for d in dims_padding]
        pad = [(ul, d-ul) for ul, d in zip(up_left_pad, dims_padding)]
        pad.append((0,0))
        return pad, np.array([pad[1][0], pad[0][0]], np.int32)

    def concatenate_images(self, map_img, legend_img, patterns_info, background_color, is_minimap, minimap_img=None):
        if type(legend_img).__name__!='NoneType':
            concatenation_args = self._randomize_concatenation_args(**self.map_concatenation_args)

            buffer = self._calc_buffer_for_img(legend_img.shape[:-1])

            legend_pos, padding4legend, coords_shift, legend_polygon = self._random_legend_placement(patterns_info, buffer, 
                                                                                        map_img.shape[:-1][::-1], 
                                                                                        legend_img.shape[:-1][::-1],
                                                                                        concatenation_args['vertical_top_pos'], 
                                                                                        concatenation_args['horizontal_left_pos'],
                                                                                        None,
                                                                                        concatenation_args['simple_concatenation'])

            
            legend_size = np.array(legend_img.shape[:-1][::-1])
            img = Image.fromarray(np.pad(map_img, padding4legend, constant_values=background_color))
            
            if concatenation_args['paste_transparent']:
                mask = Image.fromarray(self._white_background_mask(legend_img), mode='L')
                img.paste(Image.fromarray(legend_img, mode='RGB'), legend_pos, mask)
            else:
                img.paste(Image.fromarray(legend_img, mode='RGB'), legend_pos)

            if type(minimap_img).__name__!='NoneType':
                minimap_img, _ = self._resize_img(minimap_img, concatenation_args['minimap_max_dim_size'])
                minimap_size = minimap_img.size
                buffer = self._calc_buffer_for_img(minimap_size)
                minimap_pos, padding4legend, minimap_coords_shift, _ = self._random_legend_placement(patterns_info, buffer,
                                                                                            img.size, 
                                                                                            minimap_size,
                                                                                            concatenation_args['vertical_top_pos'], 
                                                                                            concatenation_args['horizontal_left_pos'], 
                                                                                            legend_polygon,
                                                                                            False)

                img = Image.fromarray(np.pad(np.array(img), padding4legend, constant_values=background_color))
                img.paste(minimap_img, minimap_pos)

                legend_pos = [d+s for d,s in zip(legend_pos, np.squeeze(minimap_coords_shift))]
                coords_shift += minimap_coords_shift

            if not is_minimap:
                img, rescale_ratio = self._resize_img(img, self.target_size)
                padding, pad_shift = self._calc_img_padding(img)

                patterns_info = self._rescale_positions(patterns_info, legend_shift=legend_pos, map_shift=np.squeeze(coords_shift), scale=rescale_ratio, pad_shift=pad_shift)

                legend_pos = (np.array(legend_pos)*rescale_ratio).astype(np.int32)+pad_shift

                legend_size = (legend_size*rescale_ratio).astype(np.int32)

                legend_label = {'pos': legend_pos, 'size': legend_size}

                if type(minimap_img).__name__!='NoneType':
                    minimap_pos = (np.array(minimap_pos)*rescale_ratio).astype(np.int32)+pad_shift
                    minimap_size = (np.array(minimap_size)*rescale_ratio).astype(np.int32)

                    minimap_label = {'pos': minimap_pos, 'size': minimap_size}

                else:
                    minimap_label = {}

                # update shapes coordinates
                pad_shift = pad_shift[np.newaxis, np.newaxis]
                if sum(np.squeeze(pad_shift))!=0:
                    for row_info in patterns_info:
                        for shape in row_info['map_args']['shapes']:
                            shape += pad_shift
                            shape = self.clockwise_points.sort_points(shape)

                img = np.pad(np.array(img), padding, constant_values=background_color)


                return img, patterns_info, legend_label, minimap_label
            else:
                return img
        else:
            # no legend and minimap
            img, rescale_ratio = self._resize_img(Image.fromarray(map_img, mode='RGB'), self.target_size)
            padding, pad_shift = self._calc_img_padding(img)

            patterns_info = self._rescale_positions(patterns_info, legend_shift=[0,0], map_shift=[0,0], scale=rescale_ratio, pad_shift=pad_shift)

            pad_shift = pad_shift[np.newaxis, np.newaxis]
            if sum(np.squeeze(pad_shift))!=0:
                for row_info in patterns_info:
                    for shape in row_info['map_args']['shapes']:
                        shape += pad_shift
                        shape = self.clockwise_points.sort_points(shape)
            
            img = np.pad(np.array(img), padding, constant_values=background_color)

            return img, patterns_info, None, None




####

######### FULL MAP GENERATOR ###########

####


class full_map_generator:
    def __init__(self, cfg, vocab, map_args_path, minimap_args_path, map_concatenation_args_path, parcel_input_args_path, bigquery_client):

        self.map_args = self._load_args(map_args_path)
        if cfg.output_type==4:
            self.map_args['random_grid_input']['single_pattern_type'] = 1
        self.minimap_args = self._load_args(minimap_args_path)
        
        map_concatenation_args = self._load_args(map_concatenation_args_path)
        map_concatenation_args['target_size'] = cfg.target_size
        self.mc = map_concatenation(map_concatenation_args)

        self.vocab = vocab

        self.map_input_gen = map_drawer_input_generator(bigquery_client, self._load_args(parcel_input_args_path), batch_size=cfg.parcel_input_batch_size,
                                                        test_mode=cfg.test_mode)

        self.map_arg_randomizer = map_drawer_arg_randomizer(**self.map_args['map_drawing_randomize_args'])
        self.minimap_arg_randomizer = map_drawer_arg_randomizer(**self.minimap_args['map_drawing_randomize_args'])

        self.target_size = cfg.target_size
        self.max_vertices_num = cfg.max_vertices_num
        self.max_shapes_num = cfg.max_shapes_num
        self.add_legend = cfg.add_legend
        self.add_minimap = cfg.add_minimap
        self.output_type = cfg.output_type

    @staticmethod
    def _load_args(path):
        with open(path, "r") as f:
            args = json.loads(f.read())
        return args

    def _gen_legend_img(self, random_grid_input, pattern_randomization_args_collection):
        # generate random legend properties
        legend_properties = gen_random_legend_properties(**random_grid_input)

        # create legend image and draw description texts
        legend_drawer = draw_legend(legend_properties, self.vocab, pattern_randomization_args_collection) # class which keeps legend image
        #legend_drawer._gen_rowwise_matrix()
        legend_drawer.draw_legend_positions()
        img = np.array(legend_drawer.img)

        patterns_info = legend_drawer.get_pattern_info()

        return img, patterns_info
    
    def reload_parcel_inputs(self,):
        self.map_input_gen.download_batch()

    def _gen_map_img(self, patterns_info, drawing_arg_randomizer, random_shapes_args, map_drawing_args, map_background_label_randomize_args, parcels_example, background_example):

        
        #if (type(parcels_example).__name__=='NoneType') | (type(background_example).__name__=='NoneType'):
        #parcels_example, background_example = next(self.map_input_gen)

        md = map_drawer(args_randomizer=drawing_arg_randomizer, 
                random_shape_drawer_args=random_shapes_args, 
                patterns_info=patterns_info,
                **parcels_example,
                background=background_example,
                map_drawing_args=map_drawing_args, 
                map_background_label_randomize_args=map_background_label_randomize_args)

        md.draw_map()

        return md.img, md.patterns_info, md.background_color
    
    
    def _shape_padding(self, shape):
        # create output tensor
        verts_to_add = max(0, self.max_vertices_num-len(shape))
        return np.concatenate([np.transpose(np.squeeze(shape[:self.max_vertices_num], axis=1), axes=[1,0]), np.zeros((2,verts_to_add), np.int32)], axis=1)
    
    def _prepare_label_for_shapes(self, patterns_info):
        return tf.constant([np.reshape(self._shape_padding(shape),(-1)) for info in patterns_info for shape in info['map_args']['shapes']][:self.max_shapes_num], tf.int32)
    
    def _prepare_edge_mask(self, patterns_info):
        edge_mask = np.zeros((self.target_size, self.target_size,1), np.uint8)
        pts = [shape for info in patterns_info for shape in info['map_args']['shapes']]
        cv.polylines(edge_mask, pts, False, 1, 1)
        return edge_mask
    
    def _gen_labels_masks(self, patterns_info):
        labels = []
        for info in patterns_info:

            if info['map_args']['shape_type']=='random_line':
                drawing_kwargs = {'thickness': 1, 'isClosed': False}  
                drawing_func = cv.polylines
            else: 
                drawing_kwargs = {}
                drawing_func = cv.fillPoly
            
            for shape in info['map_args']['shapes']:
                label = np.zeros((self.target_size, self.target_size, 1))  
                drawing_func(label, [shape], color=1, **drawing_kwargs)
                labels.append(label)

        return labels[:self.max_shapes_num]

    def gen_full_map(self, ):
        '''
            output_types:
            #0 - test mode - returns numpy map image, patterns_info and dicts for position of legend and minimap
            #1 - shapes detection - returns only tf image and tensor with vertices coordinates with format [x,x,x,0,0,...,y,y,y,0,0,...] 
            #2 - edge detection - returns map shaped image with value of 1 for edges and 0 for any other pixels
            #3 - shapes masks - returns N binary masks for N shapes
            #4 - single pattern type classification - returns class of pattern type in one-hot format
        '''
        ####################
        parcels_example, background_example = next(self.map_input_gen)

        # gen legend
        legend_img, patterns_info = self._gen_legend_img(self.map_args['random_grid_input'], self.map_args['pattern_randomization_args_collection'])

        # gen map
        map_img, patterns_info, background_color = self._gen_map_img(patterns_info,
                                                        self.map_arg_randomizer,
                                                        self.map_args['random_shapes_args'], 
                                                        self.map_args['map_drawing_args'], 
                                                        self.map_args['map_background_label_randomize_args'],
                                                        parcels_example,
                                                        background_example)
        
        # gen minimap
        if self.add_minimap:
            minimap_legend_img, minimap_patterns_info = self._gen_legend_img(self.minimap_args['random_grid_input'], self.minimap_args['pattern_randomization_args_collection'])
            minimap_img, minimap_patterns_info, minimap_background_color = self._gen_map_img(minimap_patterns_info,
                                                        self.minimap_arg_randomizer,
                                                        self.minimap_args['random_shapes_args'], 
                                                        self.minimap_args['map_drawing_args'], 
                                                        self.minimap_args['map_background_label_randomize_args'],
                                                        background_example,
                                                        background_example)
            
            minimap_img = self.mc.concatenate_images(minimap_img, minimap_legend_img, minimap_patterns_info, minimap_background_color, is_minimap=True, minimap_img=None)
        else:
            minimap_img = None

        if not self.add_legend:
            legend_img = None
        img, patterns_info, legend_label, minimap_label = self.mc.concatenate_images(map_img, legend_img, patterns_info, background_color, is_minimap=False, minimap_img=minimap_img)
        
        if self.output_type==0:
            return img, patterns_info, legend_label, minimap_label
        elif self.output_type==1:
            shape_label = self._prepare_label_for_shapes(patterns_info)
            return tf.constant(img, tf.uint8), shape_label
        elif self.output_type==2:
            edge_mask = self._prepare_edge_mask(patterns_info)
            return tf.constant(img, tf.float32)/255, tf.constant(edge_mask, tf.float32)
        elif self.output_type==3:
            labels = self._gen_labels_masks(patterns_info)
            return tf.constant(img, tf.float32)/255, tf.cast(tf.concat(labels, axis=-1), tf.bool)
        elif self.output_type==4:
            pattern_type = patterns_info[0]['pattern_style']['pattern_type']
            label = [int(pattern_type==key) for key in self.map_args['random_grid_input']['pattern_types_probs'].keys()]
            return tf.constant(img, tf.float32)/255, tf.constant(label, tf.float32)

####

######### MAP GENERATOR DECODER ###########

####

class map_generator_decoder:
    def __init__(self, cfg):
        self.target_size = cfg.target_size
        self.max_vertices_num = cfg.max_vertices_num
        self.max_shapes_num = cfg.max_shapes_num

    def decode_image(self, img):
        return tf.cast(img, tf.uint8).numpy()
    
    def decode_shape_output(self, tf_shapes):
        coords = np.expand_dims((tf.transpose(tf.reshape(tf_shapes, (-1,2,self.max_vertices_num)), perm=[0,2,1])).numpy().astype(np.int32), axis=-2)
        return [np.array([vert for vert in shape if np.all(vert)!=0]) for shape in coords]