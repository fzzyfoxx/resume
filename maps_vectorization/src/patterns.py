import numpy as np
import random
import string
from PIL import Image, ImageFont, ImageDraw
import math
import cv2 as cv

####

######### PATTERN RANDOMIZATION ###########

####

def gen_colors(light_limit=245, grayscale=False):
    if grayscale:
        color = np.repeat(np.random.randint(0,light_limit), 3, axis=0)
    else:
        while True:    
            color = np.random.randint(0,255,3)
            if len(np.where(color>light_limit))<3:
                break
    return tuple([int(x) for x in color])

class text_label_randomization:
    def __init__(self, text_label_randomize_args, pattern_size):

        self.pattern_size = pattern_size

        for key, value in text_label_randomize_args.items():
            setattr(self, key, value)

    ######### RANDOM TEXT LABEL INPUT ##########
    def random_label_text(self,):
        # random text input
        text_length = np.random.randint(*self.text_label_length_range)
        text_chars = random.choices(string.ascii_letters + string.digits, k=text_length)

        # add special character in the middle of the text
        include_special_char = np.random.binomial(1, self.special_character_prob)
        if include_special_char:
            pos = np.random.randint(1, max(2,text_length))
            text_chars.insert(pos, np.random.choice(self.special_characters))

        label_text = ''.join(text_chars)
        if np.random.binomial(1, self.all_uppercase_prob):
            label_text = label_text.upper()

        return label_text
    
    def random_text_label_args(self, ):

        width, height = self.pattern_size
        ##### random font and fontsize

        target_font_height = np.random.uniform(*self.text_label_size_rel2height_ratio_range)*height
        font_path = self.fonts_path+random.choice(self.fonts_list)
        font=ImageFont.truetype(font_path, 20)
        font_height = font.getsize('A')[1]

        font_size = int(target_font_height/font_height*20)
        font = ImageFont.truetype(font_path, font_size)
        # max width of textbox
        max_width = width*self.max_width_coverage
        while font.getsize('A'*(self.text_label_length_range[-1]+1))[0]>max_width:
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)

        font = ImageFont.truetype(font_path, font_size)

        # random stroke
        stroke = np.random.randint(*self.stroke_size_range) if np.random.binomial(1, self.use_stroke_prob) else 0

        # font color
        color = gen_colors(self.light_limit, np.random.binomial(1,self.grayscale_prob))

        # drawing position and shift from the center of the image
        x = int(round(width/2,0))
        y = int(round(height/2,0))

        if not np.random.binomial(1, self.central_position_prob):
            x += int(np.random.uniform(*self.shift_range_rel2dim)*width)
            y += int(np.random.uniform(*self.shift_range_rel2dim)*height)

        return  {
            'font': font,
            'color': color,
            'stroke': stroke,
            'pos': (x,y)
            }


class pattern_randomization:
    def __init__(self, legend_properties, striped_randomize_args, line_filled_randomize_args, line_border_randomize_args, text_label_randomize_args):
        
        # size of pattern image from legend properties
        self.pattern_size = list(map(legend_properties.get, ['pattern_width', 'pattern_height']))
        
        # parameters mapping for pattern fill
        self.pattern_args_for_randomization = {
                'solid': {},
                'striped': striped_randomize_args,
                'line_border': {},
                'line_filled': line_filled_randomize_args
            }
        
        self.pattern_randomized_args_funcs = {
            'solid': self._random_solid_pattern_args,
            'striped': self._random_striped_pattern_args,
            'line_border': self._empty_args,
            'line_filled': self._random_line_filled_pattern_args
        }
        
        # parameters for randomization of line border
        self.line_border_randomize_args = line_border_randomize_args
        # style of shared borderline
        self.common_borderline_args = self._gen_common_borderline_args()
        
        # parameters for randomization of text label on pattern
        self.text_label_randomizer = text_label_randomization(text_label_randomize_args, self.pattern_size)
        # arguments for shared text label style
        self.common_text_label_style = self.text_label_randomizer.random_text_label_args()

    def _gen_common_borderline_args(self,):
        '''
            generates random solid line border which can be shared between patterns with common_borderline parameter set to true
            returns dict with named parameters
        '''
        args = self.line_border_randomize_args.copy()
        args['no_symbols_prob'] = 1.0

        return self._random_line_border_pattern_args(**args)
    
    def _gen_single_randomized_args(self, pattern_style):
        # pattern fill arguments
        randomization_args = self.pattern_args_for_randomization[pattern_style['pattern_type']]
        randomization_func = self.pattern_randomized_args_funcs[pattern_style['pattern_type']]
        fill_args = randomization_func(**randomization_args)

        # pattern borderline arguments
        if pattern_style['any_borderline']:
            if pattern_style['use_common_borderline']:
                line_border_args = self.common_borderline_args.copy()
            else:
                line_border_args = self._random_line_border_pattern_args(**self.line_border_randomize_args)
                
            if pattern_style['pattern_type']!='line_border':
                line_border_args['pattern_lineshape_type'] = 'rectangle'
            elif pattern_style['common_borderline']:
                line_border_args['pattern_lineshape_type'] = 'line'
        else:
            line_border_args = None

        # pattern text label arguments
        if pattern_style['put_text_label']:
            text_label_style = self.text_label_randomizer.random_text_label_args() \
                                    if not pattern_style['common_text_label'] else self.common_text_label_style.copy()
            text_label_style['label_text'] = self.text_label_randomizer.random_label_text()
        else:
            text_label_style = None

        return {
            'pattern_style': pattern_style,
            'fill_args': fill_args,
            'line_border_args': line_border_args,
            'text_label_style': text_label_style
        }
    
    def gen_randomized_pattern_args(self, pattern_styles):
        self.pattern_styles = [self._gen_single_randomized_args(pattern_style) for pattern_style in pattern_styles]

    
    ######### DRAWING ARGUMENTS RANDOMIZATION FUNCTIONS ##########
    
    def _random_line_border_pattern_args(self, symbols_in_width_range, minimum_symbol_width, no_symbols_prob, no_line_prob, symbols_probs, positive_side_prob, on_line_prob, 
                                    height2width_ratio_range, spacing2width_ratio_range, filled_prob, light_limit, force_grayscale_prob, lineshape_thickness_range, solid_line_thickness_range, 
                                    solid_line_same_color_prob, horizontal_pattern_prob, horizontal_pattern_rel_position_range, pattern_padding_range):

        width = self.pattern_size[0]

        if np.random.binomial(1, no_symbols_prob):
            spacing = 0
            symbol_width = 0
            symbol_height = 0
            no_line = 0
        else:
            spacing_ratio = np.random.uniform(*spacing2width_ratio_range)

            symbols_in_width = np.random.randint(*symbols_in_width_range)
            spacing = max(math.ceil(width/symbols_in_width), math.ceil(minimum_symbol_width*(1+spacing_ratio)))

            symbol_width = math.ceil(spacing/(1+spacing_ratio))

            symbol_height = int(round(np.random.uniform(*height2width_ratio_range)*symbol_width,0))

            no_line = np.random.binomial(1, no_line_prob)

        symbol_types, symbol_probs = np.transpose(np.array([[key, symbols_probs[key]] for key in symbols_probs.keys()], dtype='object'), axes=[1,0])
        symbol_type = np.random.choice(symbol_types, p=symbol_probs.astype(np.float32))

        filled = np.random.binomial(1, filled_prob)

        grayscale = np.random.binomial(1, force_grayscale_prob)
        symbol_color = gen_colors(light_limit, grayscale)
        if np.random.binomial(1, solid_line_same_color_prob):
            line_color = symbol_color
        else:
            line_color = gen_colors(light_limit, grayscale)

        lineshape_thickness = np.random.randint(*lineshape_thickness_range)
        solid_line_thickness = np.random.randint(*solid_line_thickness_range) 
        if symbol_height>0:
            solid_line_thickness = 0 if no_line else max(1,min(int(symbol_height/2),solid_line_thickness))

        side = np.random.binomial(1, positive_side_prob)*2-1
        on_line = 1 if no_line else np.random.binomial(1,on_line_prob)

        pattern_lineshape_type = 'line' if np.random.binomial(1, horizontal_pattern_prob) else 'rectangle'
        centered = 1 if pattern_lineshape_type=='line' else 0

        horizontal_pattern_rel_position = np.random.uniform(*horizontal_pattern_rel_position_range)

        pattern_padding = np.random.randint(*pattern_padding_range)

        return {
            'spacing': spacing,
            'side': side,
            'height': symbol_height,
            'width': symbol_width,
            'filled': filled,
            'color': symbol_color,
            'shape_type': symbol_type,
            'lineshape_thickness': lineshape_thickness,
            'on_line': on_line,
            'solid_line_thickness': solid_line_thickness,
            'solid_line_color': line_color,
            'pattern_lineshape_type': pattern_lineshape_type,
            'horizontal_pattern_rel_position': horizontal_pattern_rel_position,
            'pattern_padding': pattern_padding,
            'centered': centered,
            'no_line': no_line
        }
    
    @staticmethod
    def _random_solid_pattern_args():
        return {
            'color': gen_colors()
        }
    
    @staticmethod
    def _empty_args(*args):
        return {}
    
    def _random_striped_pattern_args(self, colors_num_probs, width_on_pattern_ratio_range, same_size_prob, different_width_ratio_range, angle_range):
        width = self.pattern_size[0]
        # number of colors
        colors_num = np.random.choice(range(2,len(colors_num_probs)+2), p=colors_num_probs)
        colors = [gen_colors() for _ in range(colors_num)]


        # if all stripes should have same width
        same_size = np.random.binomial(1, same_size_prob)
        main_size = int(np.random.uniform(*width_on_pattern_ratio_range)*width)
        if same_size:
            sizes = [main_size]*colors_num
        else:
            sup_size = int(np.random.uniform(*different_width_ratio_range)*main_size)
            sizes = [main_size] + [sup_size]*(colors_num-1)
        

        angle = np.random.randint(*angle_range)

        return {
            'colors': colors,
            'sizes': sizes,
            'angle': angle
        }

    @staticmethod
    def _random_line_filled_pattern_args(light_limit, line_size_range, space_ratio_range, angle_range):
        colors = [gen_colors(light_limit), (255,255,255)]

        line_size = np.random.randint(*line_size_range)
        space_size = int(np.random.uniform(*space_ratio_range)*line_size)

        angle = np.random.randint(*angle_range)

        return {
            'colors': colors,
            'sizes': [line_size, space_size],
            'angle': angle
        }
    

####

################ PATTERN DRAWING #################

####

class drawing_patterns:
    def __init__(self, img):
        self.img = img
        self.draw = ImageDraw.Draw(self.img)

        # mapping for drawing shapes on the lines functions
        self.lineshape_functions = {
            'triangle': self._draw_triangle,
            'rectangle': self._draw_rectangle,
            'circle': self._draw_circle
        }

        self.fill_pattern_funcs = {
            'solid': self._solid_pattern,
            'striped': self._striped_pattern,
            'line_border': self._blank_pattern,
            'line_filled': self._striped_pattern
        }

        self.masks = []
    
    def set_pattern_size(self, width, height):
        self.pattern_size = (width, height)

    def draw_single_pattern(self, pts, pattern_style, xy, fill_args=None, line_border_args=None, text_label_style=None, legend_pattern=True,
                            shape_only_paste=False, transparent_paste=False):

        # generate lineshape for legend position if needed
        if (type(pts).__name__=='NoneType') & (legend_pattern==True) & (type(line_border_args).__name__!='NoneType'):
            pts, padding = self._gen_pattern_lineshape(**line_border_args)
        else:
            padding = 0

        # use pattern type to map filling function and call it with given fill_args
        self.fill_pattern_funcs[pattern_style['pattern_type']](**fill_args)

        if shape_only_paste:
            self.pattern_img = np.ones(self.pattern_size[::-1]+(3,), dtype=np.uint8)*255+(self.pattern_img+1)*self._shape_based_mask(pts)

        if padding>0:
            self._white_frame(padding)

        # if lineshape is given, draw borders
        if (type(pts).__name__!='NoneType') & (type(line_border_args).__name__!='NoneType'):
            self._draw_lines_pattern(np.squeeze(pts, axis=1), **line_border_args)

        # if parameters for text label are given then draw it
        if type(text_label_style).__name__!='NoneType':
            self._draw_text_label(**text_label_style)

        # paste drawing to main image with optional masking
        mask = None

        if transparent_paste:
            mask = self._white_background_mask()

        '''if shape_only_paste:
            shape_mask = self._shape_based_mask(pts)
            if type(mask).__name__!='NoneType':
                mask *= shape_mask
            else:
                mask = shape_mask'''

        if type(mask).__name__!='NoneType':
            self.masks.append(mask)
            mask = Image.fromarray(mask, mode='L')

        self.img.paste(Image.fromarray(self.pattern_img, mode='RGB'), xy, mask)


    def _white_background_mask(self, pattern_color=(200,200,200)):
        pattern_col_sum = np.abs(np.sum(pattern_color)-255*3)
        mask = (np.abs(np.sum(self.pattern_img.astype(np.int32), axis=-1)-255*3)/pattern_col_sum)
        mask[mask>1] = 1
        return (mask*255).astype(np.uint8)
    
    def _shape_based_mask(self, pts):
        mask = np.zeros(self.pattern_size[::-1]+(1,), np.uint8)
        cv.fillPoly(mask, [pts],(1))
        return mask
        

    ########## LINESTRINGS GENERATORS ##########

    def _gen_pattern_lineshape(self, pattern_lineshape_type, height, lineshape_thickness, solid_line_thickness,
                          horizontal_pattern_rel_position, side, on_line, **kwargs):
        
        img_width, img_height = self.pattern_size

        if (on_line) & (height>0):    
            minimal_padding = max(int(height/2), solid_line_thickness/2)
        elif (side==1) & (height>0):
            minimal_padding = math.floor(height+lineshape_thickness/2+solid_line_thickness/2)
        else:
            minimal_padding = math.floor(solid_line_thickness/2)

        if pattern_lineshape_type=='rectangle':
            x_vector = img_width-2*minimal_padding
            y_vector = img_height-2*minimal_padding
            pts = [[minimal_padding+x_vector*x, minimal_padding+y_vector*y] for x,y in zip([0,1,1,0,0],[0,0,1,1,0])]
        else:
            y = int(horizontal_pattern_rel_position*img_height)
            pts = [[x*img_width, y] for x in range(2)]
        
        return np.array(pts).reshape((-1,1,2)), minimal_padding
    
    ########## DRAW TEXT LABELS ###############

    def _draw_text_label(self, label_text, font, color, stroke, pos, **kwargs):
        img = Image.fromarray(self.pattern_img)
        draw = ImageDraw.Draw(img)
        draw.text(pos, label_text, fill=color, align='center', anchor='mm', stroke_width=stroke, stroke_fill=(np.random.randint(245,256),)*3,
                    font=font)
        
        self.pattern_img = np.array(img)

    ########## DRAW FILLED PATTERNS ###########

    def _solid_pattern(self, color=None):
        self.pattern_img = (np.ones(self.pattern_size[::-1]+(3,), np.uint8)*color).astype(np.uint8)

    def _blank_pattern(self,):
        self.pattern_img = (np.ones(self.pattern_size[::-1]+(3,), np.uint8)*255).astype(np.uint8)

    def _striped_pattern(self, colors=None, sizes=None, angle=None, **kwargs):
        
        width, height = self.pattern_size
        # draw rectangles
        iteration_size = sum(sizes)

        temp_size = int(max(width,height))*2
        temp_size = math.ceil(temp_size/iteration_size)*iteration_size

        # empty image
        pattern = np.zeros((temp_size, temp_size, 3), dtype=np.uint8)

        iterations_needed = math.ceil(temp_size/iteration_size)
        x = 0

        for i in range(iterations_needed):
            for size, color in zip(sizes, colors):
                cv.rectangle(pattern, pt1=(x,0), pt2=(x+size,temp_size), color=color, thickness=-1,)
                x += size

        # rotate
        cent = [int(iterations_needed/2*iteration_size),int(temp_size/2)]
        M = cv.getRotationMatrix2D(cent, angle, 1.0)
        pattern = cv.warpAffine(pattern, M=M, dsize=pattern.shape[:-1])

        # crop to given size
        left_crop = int((temp_size-width)/2)
        right_crop = temp_size-width-left_crop

        top_crop = int((temp_size-height)/2)
        bot_crop = temp_size-height-top_crop

        self.pattern_img = pattern[top_crop:-bot_crop, left_crop:-right_crop]

    ########## DRAW LINE PATTERNS #############

    @staticmethod
    def _sign(a):
        # sign of value with 1 for 0 value
        if a==0:
            return 1
        else:
            return int(abs(a)/a)
        
    def _white_frame(self, padding):
        return cv.rectangle(self.pattern_img, [0,0], self.pattern_img.shape[:2][::-1], color=(255,255,255), thickness=padding*2)

    def _perpendicular_point(self, p1, p2):
        '''
            Function for finding point on a perpendicular to vector p1->p2 going throught point p2
            p1, p2: start and endpoint of vector
            length: distance from generated point to point p2
            side: values[-1,1] determines on which side [left, right] of vector p1->p2 point should be generated, with point of view from p1 facing p2
        '''
        x0, y0 = p1
        x1, y1 = p2
        dx = x1-x0
        dy = y1-y0

        # calculates angle of p2 from p1 (treated like origin)
        # angle takes values between -pi to pi, negative angles are for negative p2-p1 x
        y_sign = self._sign(dy)
        x_sign = self._sign(dx)
        angle = (math.atan2(*([abs(dx), abs(dy)][::y_sign]))+math.pi*(-y_sign+1)/4)*x_sign

        # calculates an angle (new_point, p1, p2) and length of hypotenuse
        b = ((dx)**2+(dy)**2)**0.5
        c = (self.height**2+b**2)**0.5
        alpha = math.atan(self.height/b)

        # coordinates of generated point
        x = int(round(c*math.sin(angle+alpha*self.side)+x0,0))
        y = int(round(c*math.cos(angle+alpha*self.side)+y0,0))

        return np.array([x,y])
    
    def _calc_line_shift_vector(self, height_vector):
        return np.round(height_vector*self.solid_line_thickness/2/np.sum((height_vector)**2)**0.5,0).astype(np.int32)

    def _draw_triangle(self, sp, ep):
        peak = self._perpendicular_point(sp, ep)
        height_vector = peak-ep

        diffs = sp-ep
        vector_length = np.sum((diffs)**2)**0.5
        vertice_vector = diffs*self.width/vector_length/2

        if (self.solid_line_thickness>0) & (self.on_line==False):
            line_shift_vector = self._calc_line_shift_vector(height_vector)
            ep += line_shift_vector
            peak += line_shift_vector
        
        triangle_p1 = np.round(ep - vertice_vector,0).astype(np.int32)
        triangle_p2 = np.round(ep + vertice_vector,0).astype(np.int32)
        pts = np.array([triangle_p1, triangle_p2, peak]).reshape((-1,1,2))

        if self.on_line:
            pts -= np.round(height_vector/2,0).astype(np.int32).reshape((1,1,2))

        if self.filled_lineshape:
            cv.fillPoly(self.pattern_img, [pts], self.lineshape_color)
        else:
            cv.polylines(self.pattern_img, [pts], True, self.lineshape_color, thickness=self.lineshape_thickness)

    def _draw_rectangle(self, sp, ep):
        peak = self._perpendicular_point(sp, ep)

        height_vector = peak-ep

        diffs = sp-ep
        vector_length = np.sum((diffs)**2)**0.5
        vertice_vector = diffs*self.width/vector_length/2

        if (self.solid_line_thickness>0) & (self.on_line==False):
            line_shift_vector = self._calc_line_shift_vector(height_vector)
            ep += line_shift_vector
            peak += line_shift_vector

        rectangle_p1 = np.round(ep - vertice_vector,0).astype(np.int32)
        rectangle_p2 = np.round(ep + vertice_vector,0).astype(np.int32)
        rectangle_p3 = np.round(rectangle_p2+height_vector,0).astype(np.int32)
        rectangle_p4 = np.round(rectangle_p1+height_vector,0).astype(np.int32)

        pts = np.array([rectangle_p1, rectangle_p2, rectangle_p3, rectangle_p4]).reshape((-1,1,2))

        if self.on_line:
            pts -= np.round(height_vector/2,0).astype(np.int32).reshape((1,1,2))

        if self.filled_lineshape:
            cv.fillPoly(self.pattern_img, [pts], self.lineshape_color)
        else:
            cv.polylines(self.pattern_img, [pts], True, self.lineshape_color, thickness=self.lineshape_thickness)

    def _draw_circle(self, sp, ep):
        # in drawing circle height is taken for diameter and width has impact only on spacing along line
        # it is recommended to keep height and width the same for drawing circle lineshapes
        if self.on_line:
            center = np.round(ep,0).astype(np.int32)
        else:
            peak = self._perpendicular_point(sp, ep)

            if (self.solid_line_thickness>0) & (self.on_line==False):
                height_vector = peak-ep
                line_shift_vector = self._calc_line_shift_vector(height_vector)
                ep += line_shift_vector
                peak += line_shift_vector

            center = np.round(((peak-ep)/2+ep),0).astype(np.int32)

        if self.filled_lineshape:
            thickness = -1
        else:
            thickness = self.lineshape_thickness
        
        cv.circle(self.pattern_img, center, int(self.height/2), color=self.lineshape_color, thickness=thickness)

    def _draw_lines_pattern(self, linestring, spacing, side, height, width, filled, color, shape_type, centered,
                           lineshape_thickness, on_line, solid_line_thickness, solid_line_color, **kwargs):
        self.spacing = spacing
        self.side = side
        self.height = height
        self.width = width
        self.filled_lineshape = filled
        self.lineshape_color = color
        self.lineshape_function = self.lineshape_functions[shape_type]
        self.lineshape_thickness = lineshape_thickness
        self.on_line = on_line
        self.centered = centered

        self.solid_line_thickness = solid_line_thickness

        # draw solid line
        if solid_line_thickness>0:
            cv.polylines(self.pattern_img, [linestring], False, solid_line_color, thickness=solid_line_thickness, lineType=cv.LINE_8)

        # draw shapes along given linestring
        if np.all([height,width]):
            starting_space = width
            for startpoint, endpoint in zip(linestring[:-1], linestring[1:]):
                starting_space = self._draw_figures_on_line(startpoint, endpoint, starting_space)

    def _find_figure_points(self, startpoint, diffs, first_space_length):
        line_length = np.sum((diffs)**2)**0.5

        spacing_vector = diffs*self.spacing/line_length
        ep_space = max(0,line_length-self.width/2-first_space_length)
        eps = startpoint[:,np.newaxis] + first_space_length*spacing_vector[:,np.newaxis]/self.spacing + \
                        diffs[:,np.newaxis]*np.arange(0,ep_space,self.spacing)[np.newaxis]/line_length
        return np.transpose(eps, axes=[1,0])

    def _draw_figures_on_line(self, startpoint, endpoint, starting_space):
        diffs = endpoint-startpoint
        line_length = np.sum((diffs)**2)**0.5
        if np.any(diffs):
            '''spacing_vector = diffs*self.spacing/line_length
            first_space_length = max(self.width/2,(self.spacing-starting_space))
            ep_space = max(0,line_length-self.width/2-first_space_length)
            eps = startpoint[:,np.newaxis] + first_space_length*spacing_vector[:,np.newaxis]/self.spacing + \
                        diffs[:,np.newaxis]*np.arange(0,ep_space,self.spacing)[np.newaxis]/line_length
            eps = np.transpose(eps, axes=[1,0])'''
            if not self.centered:
                first_space_length = max(self.width/2,(self.spacing-starting_space))
                eps = self._find_figure_points(startpoint, diffs, first_space_length)
            else:
                half_diffs = diffs/2
                center_point = startpoint + half_diffs
                left_eps = self._find_figure_points(center_point, -half_diffs, self.spacing)[::-1]
                right_eps = self._find_figure_points(center_point, half_diffs, 0)

                eps = np.concatenate([left_eps, right_eps], axis=0)
            
            if len(eps)>0:
                for ep in eps:
                    self.lineshape_function(startpoint, ep)

                space_left = np.sum((ep-endpoint)**2)**0.5
            elif starting_space>=self.spacing:
                ep = startpoint + diffs/2
                self.lineshape_function(startpoint, ep)
                space_left = 0
            else:
                space_left = starting_space+line_length
            return space_left
        else:
            return starting_space
