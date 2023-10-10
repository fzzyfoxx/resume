import os
import sys
sys.path.append(os.path.abspath(".../"))
#os.chdir(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import spacy
from PIL import Image, ImageFont, ImageDraw
from src.patterns import pattern_randomization, gen_colors, drawing_patterns
import math


####

######### GENERATE GENERAL PARAMETERS FOR LEGEND DRAWING ###########

####

def gen_random_position_properties(pattern_type, use_common_borderline4filled_prob, put_text_label_on_pattern_prob, use_common_text_label_prob,
                                   use_any_borderline_prob):
    
    common_borderline = 1 if pattern_type=='common_borderline' else 0
    if (pattern_type in ['solid','striped','line_filled']):
        use_common_borderline = np.random.binomial(1, use_common_borderline4filled_prob)
    elif common_borderline:
        use_common_borderline = 1
    else:
        use_common_borderline = 0

    put_text_label = np.random.binomial(1, put_text_label_on_pattern_prob) if pattern_type!='common_borderline' else 0
    common_text_label = np.random.binomial(1, use_common_text_label_prob) if (pattern_type!='common_borderline') & (put_text_label) else 0

    if pattern_type=='common_borderline':
        pattern_type = 'line_border'

    any_borderline = np.random.binomial(1, use_any_borderline_prob) if pattern_type=='line_filled' else 1

    return {
        'pattern_type': pattern_type,
        'use_common_borderline': use_common_borderline,
        'put_text_label': put_text_label,
        'common_borderline': common_borderline,
        'common_text_label': common_text_label,
        'any_borderline': any_borderline
    }


def gen_random_legend_properties(
        legend_positions_range,
        single_pattern_type,
        pattern_types_probs,
        center_oversized_description_prob,
        center_fitting_description_prob,
        use_common_borderline4filled_prob,
        use_any_borderline_prob,
        put_text_label_on_pattern_prob,
        use_common_text_label_prob,
        show_common_borderline_on_pattern_prob,
        height2width_ratio_range,
        patterns_size_range,
        no_space_between_prob,
        vertical_space_size_range,
        horizontal_space_size_range,
        include_headlines_prob,
        same_fontsize4headline_prob,
        headline_fontsize_add_range,
        headline_size_distr_args,
        space_after_headline_range,
        headline_padding_prob,
        interline_size_range,
        textbox_width_ratio_range,
        description_words_distr_args,
        font_size_range,
        add_stroke_prob,
        stroke_size_range,
        uppercase_prob,
        fonts_path,
        fonts_list,
        one_column_prob,
        max_columns_num_range,
        space_between_columns_range,
        rowwise_prob,
        legend_vertical_padding_range,
        legend_horizontal_padding_range,
        frames_around_patterns_prob,
        frames_around_patterns_thickness_range,
        target_max_dim_size
        ):
    
    # number of positions
    legend_positions = np.random.randint(*legend_positions_range)

    # show common borderline as additional position
    separate_pattern4common_borderline = np.random.binomial(1, show_common_borderline_on_pattern_prob)

    # generate random patterns
    pattern_types, pattern_probs = np.transpose(np.array([[key, pattern_types_probs[key]] for key in pattern_types_probs.keys()], dtype='object'), axes=[1,0])
    if single_pattern_type:
        patterns_order = np.repeat(np.random.choice(pattern_types, size=1, p=pattern_probs.astype(np.float32)), legend_positions, axis=0)
    else:
        patterns_order = np.random.choice(pattern_types, size=legend_positions, p=pattern_probs.astype(np.float32))

    # add common borderline pattern
    if separate_pattern4common_borderline:
        patterns_order = np.append(patterns_order, 'common_borderline')
        legend_positions += 1
    np.random.shuffle(patterns_order)
    
    # additional parameters for position
    patterns_order = [gen_random_position_properties(x, use_common_borderline4filled_prob, put_text_label_on_pattern_prob,
                                                     use_common_text_label_prob, use_any_borderline_prob) for x in patterns_order]

    # number of description words for position
    description_words = np.round(np.random.gamma(*description_words_distr_args, size=legend_positions),0).astype(np.int32)+1

    # divide positions into headlines groups
    if np.random.binomial(1, include_headlines_prob):
        headlines = True
        headline_sizes = []
        positions_left = legend_positions
        positions_taken = 0
        while True:
            headline_size = int(round(min(max(np.random.normal(*headline_size_distr_args),1), positions_left),0))
            headline_sizes.append(headline_size+positions_taken)
            positions_left -= headline_size
            positions_taken += headline_size
            if positions_left<=0:
                break
        patterns_order = np.split(patterns_order, headline_sizes[:-1])
        description_words = np.split(description_words, headline_sizes[:-1])
    else:
        headlines = False
        patterns_order = [patterns_order]
        description_words = [description_words]

    # number of headlines words
    if headlines:
        headlines_words = np.round(np.random.gamma(*description_words_distr_args, size=len(description_words)),0).astype(np.int32)+1
        headline_padding = np.random.binomial(1,headline_padding_prob)
    else:
        headlines_words = None
        headline_padding = None

    # space after headline
    space_after_headline = np.random.randint(*space_after_headline_range)

    # interline size
    interline_size = round(np.random.uniform(*interline_size_range),1)

    # img_size
    pattern_width = np.random.randint(*patterns_size_range)
    pattern_height = int(np.random.uniform(*height2width_ratio_range)*pattern_width)

    # frame around patterns
    frames_around_patterns_thickness = np.random.randint(*frames_around_patterns_thickness_range) if np.random.binomial(1, frames_around_patterns_prob) else 0

    # horizontal space between pattern and description
    patterns_horizontal_space = np.random.randint(*horizontal_space_size_range) + frames_around_patterns_thickness

    # vertical space between images
    if np.random.binomial(1, no_space_between_prob):
        patterns_vertical_space = 0
    else:
        patterns_vertical_space = np.random.randint(*vertical_space_size_range)
    patterns_vertical_space += frames_around_patterns_thickness

    
    # textbox width
    textbox_width = int(np.random.uniform(*textbox_width_ratio_range)*pattern_width)

    # font size
    font_size = np.random.randint(*font_size_range)

    # headline font size
    if headlines:
        if np.random.binomial(1, same_fontsize4headline_prob):
            headline_font_size = font_size
        else:
            headline_font_size = font_size + np.random.randint(*headline_fontsize_add_range)
    else:
        headline_font_size = None

    # uppercase
    uppercase = bool(np.random.binomial(1,uppercase_prob))

    # random font
    font_path = fonts_path + np.random.choice(fonts_list)

    # draw row-wise or column-wise
    rowwise = bool(np.random.binomial(1, rowwise_prob))

    # maximum number of columns
    if np.random.binomial(1, one_column_prob) & rowwise:
        max_columns_num = 1
    else:
        max_columns_num = min(legend_positions, np.random.randint(*max_columns_num_range))

    # stroke size
    if np.random.binomial(1, add_stroke_prob):
        stroke_width = np.random.randint(*stroke_size_range)
    else:
        stroke_width = 0

    

    # description adjust methods
    center_oversized_description = np.random.binomial(1, center_oversized_description_prob)
    center_fitting_description = np.random.binomial(1, center_fitting_description_prob)

    # space between columns
    space_between_columns = np.random.randint(*space_between_columns_range)

    # legend padding
    legend_vertical_padding = np.random.randint(*legend_vertical_padding_range)
    legend_horizontal_padding = np.random.randint(*legend_horizontal_padding_range)

    return {
        'legend_positions': legend_positions,
        'patterns_order': patterns_order,
        'separate_pattern4common_borderline': separate_pattern4common_borderline,
        'description_words': description_words,
        'pattern_width': pattern_width,
        'pattern_height': pattern_height,
        'patterns_vertical_space': patterns_vertical_space,
        'patterns_horizontal_space': patterns_horizontal_space,
        'textbox_width': textbox_width,
        'headlines': headlines,
        'headlines_words': headlines_words,
        'space_after_headline': space_after_headline,
        'headline_padding': headline_padding,
        'interline_size': interline_size,
        'font_size': font_size,
        'headline_font_size': headline_font_size,
        'stroke_width': stroke_width,
        'uppercase': uppercase,
        'font_path': font_path,
        'max_columns_num': max_columns_num,
        'space_between_columns': space_between_columns,
        'rowwise': rowwise,
        'center_oversized_description': center_oversized_description,
        'center_fitting_description': center_fitting_description,
        'legend_vertical_padding': legend_vertical_padding,
        'legend_horizontal_padding': legend_horizontal_padding,
        'frames_around_patterns_thickness': frames_around_patterns_thickness,
        'target_max_dim_size': target_max_dim_size
            }



####

######### DRAW LEGEND CLASS ###########

####




def load_vocab():
    nlp = spacy.load("pl_core_news_sm")
    tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)
    vocab = list(filter(lambda x: len(x)<15,list(set(nlp.vocab.strings))))
    vocab = [token.text for doc in tokenizer.pipe(vocab) for token in doc if token.is_alpha]

    return vocab

class draw_legend:
    def __init__(self, legend_properties, vocab, pattern_randomization_args_collection):
        self.args = legend_properties

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.font = ImageFont.truetype(self.args['font_path'], self.args['font_size'])
        self.headline_font = ImageFont.truetype(self.args['font_path'], self.args['headline_font_size']) if self.args['headline_font_size'] else self.font

        self.box_width = self.args['pattern_width']+self.args['patterns_horizontal_space']+self.args['textbox_width']
        self.text_inbox_pos = self.args['pattern_width']+self.args['patterns_horizontal_space']

        self.pattern_size = (self.args['pattern_width'], self.args['pattern_height'])

        self._generate_descriptions()

        self.pattern_randomizer = pattern_randomization(legend_properties, **pattern_randomization_args_collection)

    @staticmethod
    def _get_font_size(font, text, spacing=1):
        '''bbox = font.getbbox(text, spacing=spacing)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])'''
        return font.getsize_multiline(text, spacing=spacing)

    def _generate_textbox(self, words_count, text_type, max_textbox_width, pattern_style):
        
        font = self.font if text_type=='description' else self.headline_font
        
        words = [self.vocab[i] for i in np.random.randint(0,self.vocab_size, words_count)]
        words_width = [self._get_font_size(font, t+' ')[0] for t in words]

        rows_collection = []
        rows_widths = []
        current_row = []
        current_row_width = 0

        for width, word in zip(words_width, words):
            if width<=max_textbox_width:
                if (current_row_width+width)>max_textbox_width:
                    rows_collection.append(current_row)
                    rows_widths.append(current_row_width)
                    current_row = [word]
                    current_row_width = width
                else:
                    current_row.append(word)
                    current_row_width += width
        if len(current_row)>0:
            rows_collection.append(current_row)
            rows_widths.append(current_row_width)

        sentence = '\n'.join([' '.join(x) for x in rows_collection])
        if self.args['uppercase']:
            sentence = sentence.upper()

        
        textbox_size = [int(x) for x in self._get_font_size(font, sentence, spacing=self.args['interline_size'])] #font.getsize_multiline(sentence, spacing=self.args['interline_size'])

        cell_height = max(textbox_size[1] + (self.args['space_after_headline'] if text_type=='headline' else self.args['patterns_vertical_space']), 
                          ((self.args['pattern_height'] + self.args['patterns_vertical_space']) if text_type=='description' else 0))

        return {
            'text': sentence,
            'textbox_size': textbox_size,
            'text_type': text_type,
            'cell_height': cell_height,
            'pattern_style': pattern_style
        }

    
    @staticmethod
    def _combine_descriptions(descriptions, headline):
        if headline:
            return headline+descriptions
        else:
            return descriptions
    
    def _generate_descriptions(self,):
        descriptions = [[self._generate_textbox(n, 'description', self.args['textbox_width'], np) for n, np in zip(h,hp)] for h, hp in zip(self.args['description_words'], self.args['patterns_order'])]

        if self.args['headlines']:
            additional_space = (self.args['headline_padding']-1)*(self.args['pattern_width']+self.args['patterns_vertical_space'])
            headline_descriptions = [[self._generate_textbox(n, 'headline', self.args['textbox_width']+additional_space, None)] 
                                     for n in self.args['headlines_words']]
        else:
            headline_descriptions = [[None]]*len(descriptions)
        
        self.text_rows = [r for d,h in zip(descriptions, headline_descriptions) for r in self._combine_descriptions(d,h) if r!=None]
        
    
    @staticmethod
    def _check_size_range(sizes):
        sizes_sums = [sum(c) for c in sizes]
        return np.max(sizes_sums)-np.min(sizes_sums)

    @staticmethod
    def _move_to_right(sizes):
        if len(sizes)>1:
            for i in range(len(sizes)-1):
                left_col = sizes[i]
                right_col = sizes[i+1]
                left_sum = sum(left_col)
                right_sum = sum(right_col)
                if len(left_col)>0:
                    passed_elem_size = left_col[-1]
                    if (left_sum>right_sum) & (len(left_col)>1) & ((left_sum-right_sum)>abs(right_sum+passed_elem_size*2-left_sum)):
                        right_col.insert(0,passed_elem_size)
                        left_col.pop(-1)
        return sizes
    
    def _gen_rowwise_matrix(self,):
        positions_num = len(self.text_rows)

        current_column = []
        columns_collection = []
        col_num = 1
        i = 0
        current_height = 0
        approx_row_height = int(sum([r['cell_height'] for r in self.text_rows])/self.args['max_columns_num'])

        while i<positions_num: 
            _, _, text_type, cell_height, _ = self.text_rows[i].values()
            if text_type=='headline':
                i += 1
                cell_height += self.text_rows[i]['cell_height']

            current_height += cell_height

            current_column.append(cell_height)

            if (col_num<=self.args['max_columns_num']) & (current_height>=approx_row_height):
                columns_collection.append(current_column)
                current_column = []
                current_height = 0
                col_num += 1
            i += 1

        if len(current_column)>0:
            columns_collection.append(current_column)
        
        col_diff = self.args['max_columns_num']-len(columns_collection)
        for i in range(col_diff):
            columns_collection.append([])


        current_range = self._check_size_range(columns_collection)
        previous_range = current_range+1

        while current_range<previous_range:
            columns_collection = self._move_to_right(columns_collection)
            previous_range = current_range
            current_range = self._check_size_range(columns_collection)

        # set upper left corner of position box including pattern image
        x = self.args['legend_horizontal_padding']
        y = self.args['legend_vertical_padding']
        col = 0
        row = 0
        headline_height = 0
        for text_row in self.text_rows:
            _, textbox_size, text_type, cell_height, _ = text_row.values()

            #text_row['yx'] = (y,x)
            text_row['pattern_xy'], text_row['text_xy'] = self._adjust_position_corners(x,y, textbox_size, text_type)
            text_row['bot_right'] = (y+cell_height, x+self.box_width)

            if text_type=='headline':
                y += cell_height
                headline_height = cell_height
            else:
                current_col = columns_collection[col]
                if len(current_col)==row+1:
                    col += 1
                    row = 0
                    y = self.args['legend_vertical_padding']
                    x += self.box_width + self.args['space_between_columns']
                else:
                    y += current_col[row]-headline_height
                    row += 1
                headline_height = 0

    def _gen_columnwise_matrix(self, ):

        col_num = self.args['max_columns_num']-1
        starting_x = self.args['legend_horizontal_padding']
        x = self.args['legend_horizontal_padding']
        y = self.args['legend_vertical_padding']
        col = 0
        row_height = 0

        for text_row in self.text_rows:
            _, textbox_size, text_type, cell_height, _ = text_row.values()

            if (text_type=='headline') & (col!=0):
                y += row_height
                x = starting_x
                col = 0

            text_row['pattern_xy'], text_row['text_xy'] = self._adjust_position_corners(x,y, textbox_size, text_type)
            text_row['bot_right'] = (y+cell_height, x+self.box_width)

            row_height = max(row_height, cell_height)

            if (text_type=='headline') | (col==col_num):
                y += row_height
                x = starting_x
                if col==col_num:
                    col = 0
                row_height = 0
            else:
                x += self.box_width + self.args['space_between_columns']
                col += 1


    def _adjust_position_corners(self, x, y, textbox_size, text_type):
        _, pattern_height = self.pattern_size
        _, text_height = textbox_size

        if text_type=='description':
            pattern_y = y+int((text_height-pattern_height)/2) if (text_height>pattern_height) & self.args['center_oversized_description'] else y
            pattern_xy = (x, pattern_y)
        else:
            pattern_xy = None

        if text_type=='description':
            text_y = y+int((pattern_height-text_height)/2) if (pattern_height>text_height) & self.args['center_fitting_description'] else y
        else:
            text_y = y
        text_x = x + (self.text_inbox_pos if (text_type=='description') | (self.args['headline_padding']!=None) else 0)

        return pattern_xy, (text_x, text_y)
    
    def _gen_pattern_frame_args(self, ):
        frame_thickness = self.args['frames_around_patterns_thickness']
        outline = gen_colors(light_limit=150, grayscale=True)

        self.pattern_frame_args = {'outline': outline, 'width': frame_thickness}

    def _draw_pattern_frame(self, xy):
        bot_right = tuple([a+b+math.ceil(self.args['frames_around_patterns_thickness']/2) for a,b in zip(xy, self.pattern_size)])
        top_left = tuple([a-math.floor(self.args['frames_around_patterns_thickness']/2) for a in xy])

        self.draw.rectangle([top_left,bot_right], **self.pattern_frame_args)

    
    def _calc_legend_size(self, text_rows):
        return (np.max([r['bot_right'] for r in text_rows], axis=0)+np.array([self.args['legend_vertical_padding'],self.args['legend_horizontal_padding']])).tolist()

    def _resize_to_target(self, ):
        leg_width, leg_height = self.img.size
        resize_ratio = self.args['target_max_dim_size']/max(leg_width, leg_height)
        new_width = int(resize_ratio*leg_width)
        new_height = int(resize_ratio*leg_height)

        self.img = self.img.resize((new_width,new_height))
        
    def draw_legend_positions(self,):
        if self.args['rowwise']:
            self._gen_rowwise_matrix()
        else:
            self._gen_columnwise_matrix()
        self._gen_pattern_frame_args()

        self.img = Image.fromarray(np.ones(self._calc_legend_size(self.text_rows)+[3], dtype=np.uint8)*255)
        self.draw = ImageDraw.Draw(self.img)

        self._gen_pattern_styles()
        style_iter = iter(self.pattern_randomizer.pattern_styles)

        pattern_drawer = drawing_patterns(self.img)
        pattern_drawer.set_pattern_size(self.args['pattern_width'],self.args['pattern_height'])

        for text_row in self.text_rows:
            sentence, _, text_type, _, pattern_style, pattern_xy, text_xy, _ = text_row.values()
            # draw text
            font = self.font if text_type=='description' else self.headline_font
            
            self.draw.text(text_xy, sentence, font=font, fill=(0,0,0), anchor='la', 
                      stroke_width=self.args['stroke_width'], stroke_fill=(255,255,255), spacing=self.args['interline_size'])
            
            if text_type=='description':
                pattern_style, fill_args, line_border_args, text_label_style  = next(style_iter).values()
                passed_line_border_args = None if self.args['separate_pattern4common_borderline'] & (not pattern_style['common_borderline']) \
                                            & pattern_style['use_common_borderline'] else line_border_args
                
                if self.args['frames_around_patterns_thickness']>0:
                    self._draw_pattern_frame(pattern_xy)

                pattern_drawer.draw_single_pattern(None, pattern_style, pattern_xy, fill_args, passed_line_border_args,
                                                            text_label_style, legend_pattern=True, shape_only_paste=False, transparent_paste=False)
                
        #self._resize_to_target()
    
    def _gen_pattern_styles(self, ):
        self.pattern_randomizer.gen_randomized_pattern_args([r['pattern_style'] for r in self.text_rows if r['text_type']=='description'])

    def get_pattern_info(self, ):
        drawing_positions = [x for x in self.text_rows if x['text_type']=='description']
        for pos, pattern_style in zip(drawing_positions, self.pattern_randomizer.pattern_styles):
            for key in pattern_style.keys():
                pos[key] = pattern_style[key]

        return [x for x in drawing_positions if x['pattern_style']['common_borderline']==0]