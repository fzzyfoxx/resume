import sys
import json
from IPython.display import clear_output

from models_src.VecDataset import MultishapeMapGenerator, DatasetGenerator, blur_img, op_k_shape_points_vecs
from exp_lib.utils.cfg_loader import cfg_loader

ds_cfg_path = sys.argv[1] 

cfg = cfg_loader(ds_cfg_path)

cfg.sample_points = cfg.sample_shapes * cfg.shape_samples

with open(cfg.msmg_config_path) as json_msmg:
    msmg_config = json.load(json_msmg)

msmg = MultishapeMapGenerator(**msmg_config['msmg'])

dg = DatasetGenerator(map_generator=msmg,
                      ds_path=cfg.ds_path,
                      **msmg_config['dg'],
                      parallel_calls=4, 
                      padded_batch=True, 
                      output_filter=['img','vecs_masks', 'bbox_masks', 'vecs', 'bboxes', 'vecs_mask', 'bbox_mask', 'shape_thickness'],
                      preprocess_funcs=[(blur_img, {'blur_ratio_range': (0.1, 0.6), 'kernel_size': 3, 'color_rand_range': 0.1}, True),
                                        (op_k_shape_points_vecs, 
                                        {
                                            'n': cfg.sample_shapes, 
                                            'k': cfg.shape_samples, 
                                            'vec_label': cfg.vec_label, 
                                            'add_thickness': cfg.add_thickness, 
                                            'add_probe_label': cfg.add_probe_label
                                        }, 
                                        True)]
                      )

ds, train_steps = dg.dataset(from_saved=True, batch_size=cfg.train_batch_size, validation=False, val_idxs=cfg.val_idxs, shuffle_buffer_size=cfg.shuffle_buffer_size)
val_ds, val_steps = dg.dataset(from_saved=True, batch_size=cfg.val_batch_size, validation=True, val_idxs=cfg.val_idxs)
test_ds, test_steps = dg.dataset(from_saved=True, batch_size=cfg.test_batch_size, validation=True, val_idxs=cfg.val_idxs)

ds_iter = iter(ds)
val_iter = iter(val_ds)
test_iter = iter(test_ds)

clear_output(wait=True)