import sys
import json
from IPython.display import clear_output
import argparse

from models_src.VecDataset import MultishapeMapGenerator, DatasetGenerator, blur_img, op_rotated_enc_full_label, colors_random_shift
from exp_lib.utils.cfg_loader import cfg_loader

parser = argparse.ArgumentParser()

parser.add_argument("--rotated_bbox_label", default=None, type=int, help="op_rotated_enc_full_label rotated_bbox_label argument")
parser.add_argument("--angle_input_rand_range", default=None, type=int, help="op_rotated_enc_full_label angle_input_rand_range argument")
parser.add_argument("--val_color_shift", default=0, type=int, help="if 1 then color shift preprocessing is also applied to validation dataset")


kwargs, args = parser.parse_known_args()
ds_cfg_path = args[0] 
rotated_bbox_label = kwargs.rotated_bbox_label
angle_input_rand_range = kwargs.angle_input_rand_range
val_color_shift = bool(kwargs.val_color_shift)

cfg = cfg_loader(ds_cfg_path)

if rotated_bbox_label is not None:
    cfg.rotated_bbox_label = bool(rotated_bbox_label)

if angle_input_rand_range is not None:
    cfg.angle_input_rand_range = angle_input_rand_range

if not hasattr(cfg, 'splitted_conf'):
    cfg.splitted_conf = False

with open(cfg.msmg_config_path) as json_msmg:
    msmg_config = json.load(json_msmg)

msmg = MultishapeMapGenerator(**msmg_config['msmg'])

dg = DatasetGenerator(map_generator=msmg,
                      ds_path=cfg.ds_path,
                      **msmg_config['dg'],
                      parallel_calls=4, 
                      padded_batch=True, 
                      output_filter=['img', 'vecs', 'bboxes', 'vecs_mask', 'bbox_mask', 'vecs_masks', 'bbox_masks', 'line_label', 'shape_label', 'thickness_label'],
                      preprocess_funcs=[(colors_random_shift, {}, val_color_shift),
                                        (blur_img, {'blur_ratio_range': (0.1, 0.6), 'kernel_size': 3, 'color_rand_range': 0.1}, True),
                                        (op_rotated_enc_full_label, {'angle_samples_num': cfg.angle_samples_num, 'random_angle_weight': cfg.random_angle_weight, 'rotated_bbox_label': cfg.rotated_bbox_label, 'angle_input_rand_range': cfg.angle_input_rand_range, 'splitted_conf': cfg.splitted_conf}, True)]
                      )

ds, train_steps = dg.dataset(from_saved=True, batch_size=cfg.train_batch_size, validation=False, val_idxs=cfg.val_idxs, shuffle_buffer_size=cfg.shuffle_buffer_size)
val_ds, val_steps = dg.dataset(from_saved=True, batch_size=cfg.val_batch_size, validation=True, val_idxs=cfg.val_idxs)
test_ds, test_steps = dg.dataset(from_saved=True, batch_size=cfg.test_batch_size, validation=True, val_idxs=cfg.val_idxs)

ds_iter = iter(ds)
val_iter = iter(val_ds)
test_iter = iter(test_ds)

clear_output(wait=True)