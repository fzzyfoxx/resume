import sys
import json
from IPython.display import clear_output
import argparse

from models_src.VecDataset import MultishapeMapGenerator, DatasetGenerator, blur_img, colors_random_shift
from models_src.NoGame import op_pattern_vertices, get_max_pattern_shapes_num
from exp_lib.utils.cfg_loader import cfg_loader

parser = argparse.ArgumentParser()

parser.add_argument("--val_color_shift", default=0, type=int, help="if 1 then color shift preprocessing is also applied to validation dataset")
parser.add_argument("--min_split_length", default=0.0, type=float, help="If 0 then the parameter is taken from cfg, otherwise it is used as min_split_length")

kwargs, args = parser.parse_known_args()
ds_cfg_path = args[0] 
val_color_shift = bool(kwargs.val_color_shift)
min_split_length = float(kwargs.min_split_length)

cfg = cfg_loader(ds_cfg_path)

if min_split_length != 0.0:
    cfg.min_split_length = min_split_length

with open(cfg.msmg_config_path) as json_msmg:
    msmg_config = json.load(json_msmg)

msmg = MultishapeMapGenerator(**msmg_config['msmg'])

size = msmg_config['msmg']['size']
max_shapes_num = get_max_pattern_shapes_num(msmg_config['msmg'])

dg = DatasetGenerator(map_generator=msmg,
                      ds_path=cfg.ds_path,
                      **msmg_config['dg'],
                      parallel_calls=4, 
                      padded_batch=True, 
                      output_filter=['img','vecs', 'bboxes', 'pattern_masks', 'vecs_pattern_idxs', 'bbox_pattern_idxs'],
                      preprocess_funcs=[(colors_random_shift, {}, val_color_shift),
                                        (blur_img, {'blur_ratio_range': (0.1, 0.6), 'kernel_size': 3, 'color_rand_range': 0.1}, True),
                                        (op_pattern_vertices, {'size': size, 'min_split_length': cfg.min_split_length, 'max_shapes_num': max_shapes_num}, True)]
                      )

ds, train_steps = dg.dataset(from_saved=True, batch_size=cfg.train_batch_size, validation=False, val_idxs=cfg.val_idxs, shuffle_buffer_size=cfg.shuffle_buffer_size)
val_ds, val_steps = dg.dataset(from_saved=True, batch_size=cfg.val_batch_size, validation=True, val_idxs=cfg.val_idxs)
test_ds, test_steps = dg.dataset(from_saved=True, batch_size=cfg.test_batch_size, validation=True, val_idxs=cfg.val_idxs)

ds_iter = iter(ds)
val_iter = iter(val_ds)
test_iter = iter(test_ds)

clear_output(wait=True)