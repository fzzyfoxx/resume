import sys
import json
from IPython.display import clear_output
import argparse

from models_src.VecDataset import MultishapeMapGenerator, DatasetGenerator, blur_img, op_pixel_similarity_shapes, colors_random_shift
from exp_lib.utils.cfg_loader import cfg_loader

parser = argparse.ArgumentParser()

parser.add_argument("--val_color_shift", default=0, type=int, help="if 1 then color shift preprocessing is also applied to validation dataset")

kwargs, args = parser.parse_known_args()
ds_cfg_path = args[0] 
val_color_shift = bool(kwargs.val_color_shift)

cfg = cfg_loader(ds_cfg_path)

with open(cfg.msmg_config_path) as json_msmg:
    msmg_config = json.load(json_msmg)

msmg = MultishapeMapGenerator(**msmg_config['msmg'])

dg = DatasetGenerator(map_generator=msmg,
                      ds_path=cfg.ds_path,
                      **msmg_config['dg'],
                      parallel_calls=4, 
                      padded_batch=True, 
                      output_filter=['img','pattern_masks', 'shape_masks'],
                      preprocess_funcs=[(colors_random_shift, {}, val_color_shift),
                                        (blur_img, {'blur_ratio_range': (0.1, 0.6), 'kernel_size': 3, 'color_rand_range': 0.1}, True),
                                        (op_pixel_similarity_shapes, {}, True)]
                      )

ds, train_steps = dg.dataset(from_saved=True, batch_size=cfg.train_batch_size, validation=False, val_idxs=cfg.val_idxs, shuffle_buffer_size=cfg.shuffle_buffer_size)
val_ds, val_steps = dg.dataset(from_saved=True, batch_size=cfg.val_batch_size, validation=True, val_idxs=cfg.val_idxs)
test_ds, test_steps = dg.dataset(from_saved=True, batch_size=cfg.test_batch_size, validation=True, val_idxs=cfg.val_idxs)

ds_iter = iter(ds)
val_iter = iter(val_ds)
test_iter = iter(test_ds)

clear_output(wait=True)