from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from models_src.Metrics import WeightedF12D
from models_src.fft_lib import xy_coords
from IPython.display import clear_output
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--s", default=5, type=int, help="Number of columns/images input")
parser.add_argument("--next", default=1, type=int, help="Prepare plots for another batch")
parser.add_argument("--offset", default=0, type=int, help="Batch starting index")

kwargs = vars(parser.parse_args())

s = kwargs['s']
next_batch = kwargs['next']
offset = kwargs['offset']

if next_batch:
    img, labels, weights = next(test_iter) # type: ignore

height = img.shape[1]
width = img.shape[2]
f1_loss = WeightedF12D()

shape_class, angle, thickness, center_vec = labels.values()
_, angle_mask, thickness_mask, center_vec_mask = [tf.where(elem>0, 1.0, 0.0) for elem in weights.values()]

background_mask, lines_mask, bbox_mask = tf.split(shape_class, 3, axis=-1)

pred_shape_class, pred_angle, pred_thickness, pred_center_vec = trainer.model(img, training=False).values() # type: ignore
pred_background_mask, pred_lines_mask, pred_bbox_mask = tf.split(pred_shape_class, 3, axis=-1)

pred_thickness *= thickness_mask
#pred_center_vec *= center_vec_mask
#pred_angle *= angle_mask

clear_output()
print([f1_loss(label, pred).numpy() for label, pred in zip([background_mask, lines_mask, bbox_mask], [pred_background_mask, pred_lines_mask, pred_bbox_mask])])

x, y = tf.split(xy_coords((height,width)), 2, axis=-1)

fig, axs = plt.subplots(11, s, figsize=(s*3, 11*3))

for n in range(s):
    for i, plot_img in enumerate([img, lines_mask, pred_lines_mask, bbox_mask, pred_bbox_mask, thickness, pred_thickness]):
        axs[i,n].imshow(plot_img[n+offset])
        axs[i,n].set_ylim(0, height-1)

    for i, plot_angle in enumerate([angle*angle_mask[n+offset], pred_angle]):
        ay = tf.squeeze(tf.sin(plot_angle[n+offset]), axis=-1)
        ax = tf.squeeze(tf.cos(plot_angle[n+offset]), axis=-1)
        axs[i+7,n].imshow(angle_mask[n+offset], cmap='gray')
        axs[i+7,n].quiver(x, y, ax, ay, color='red', width=0.003, scale=40)
        axs[i+7,n].set_ylim(0, height-1)

    for i, plot_vec in enumerate([center_vec, pred_center_vec]):

        axs[i+9,n].imshow(center_vec_mask[n+offset], cmap='gray')
        axs[i+9,n].quiver(x, y, plot_vec[n+offset,...,1], plot_vec[n+offset,...,0], color='red', width=0.003, scale=40)
        axs[i+9,n].set_ylim(0, height-1)


for i, title in enumerate(['img', 'lines_mask', 'pred_lines_mask', 'bbox_mask', 'pred_bbox_mask', 'thickness', 'pred_thickness', 'angle', 'pred_angle', 'center_vec', 'pred_center_vec']):
    axs[i,0].set_ylabel(title, rotation=90, fontsize=12)

plt.show()