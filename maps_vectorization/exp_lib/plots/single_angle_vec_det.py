import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from models_src.VecModels import prepare_vec_label2plot
from models_src.VecModels import NoSplitMixedBboxVecMultiPropMetric, gen_rot_matrix_yx
from IPython.display import clear_output
from exp_lib.plots.plot_utils import bold_argmax_print, create_bbox_from_vector
import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument("--s", default=5, type=int, help="Number of columns/images input")
parser.add_argument("--r", default=8, type=int, help="Number of rows/sample points")
parser.add_argument("--next", default=1, type=int, help="Prepare plots for another batch")

kwargs = vars(parser.parse_args())

num_examples_per_image = kwargs['s']
r = kwargs['r']
next_batch = kwargs['next']


if next_batch:
    print('Processing next batch')
    inputs, labels, weights = next(test_iter) # type: ignore
    img, angle_input = inputs['img'], inputs['angle_input']
    vec_label, conf_label, shape_class, thickness_label = labels['vecs'], labels['conf'], labels['class'], labels['thickness']
    vec_weights, conf_weights, class_weights, thickness_weights = weights['vecs'], weights['conf'], weights['class'], weights['thickness']

    preds = trainer.model(inputs, training=False) # type: ignore
    vec_pred, conf_pred, class_pred, thickness_pred = preds['vecs'], preds['conf'], preds['class'], preds['thickness']

img_height = img.shape[1]

# Create a figure for plotting
fig, axs = plt.subplots(r, num_examples_per_image+2, figsize=((num_examples_per_image+2) * 2, r * 2))

# Iterate over the number of rows (images)
for row in range(r):
    # Iterate over the number of examples per image
    a_y = tf.sin(angle_input[row])
    a_x = tf.cos(angle_input[row])
    cx, cy = (img.shape[2]-1) / 2, (img.shape[1]-1) / 2

    ax = axs[row, 0]
    axs[row, 0].imshow(conf_label[row])
    axs[row, 1].imshow(conf_pred[row], vmin=0, vmax=1)

    for i, title in enumerate(['conf_label', 'conf_pred']):
        axs[row, i].set_title(title, fontsize=7)
        axs[row, i].set_ylim(0.5, img_height-0.5)
        axs[row, i].set_xticks([])
        axs[row, i].set_yticks([])

    for col in range(num_examples_per_image):
        # Select random points where vec_weights > 0
        valid_points = tf.where(vec_weights[row] > 0)
        selected_point = random.choice(valid_points.numpy())
        y, x = selected_point[0], selected_point[1]
        
        # Get angle
        angle = angle_input[row, 0]
        
        a_y = tf.sin(angle)
        a_x = tf.cos(angle)
        cx, cy = (img.shape[2]-1) / 2, (img.shape[1]-1) / 2

        # Extract the vector label for the selected point
        vec = vec_label[row, y, x]
        class_idx = tf.argmax(shape_class[row, y, x])
        ex_pred_class = class_pred[row, y, x]   
        pred_class_idx = tf.argmax(ex_pred_class)

        pred_vec = vec_pred[row, y, x]
        if pred_class_idx==1:
            rot_matrix = gen_rot_matrix_yx(angle)
            inv_rot_matrix = gen_rot_matrix_yx(-angle)
            #vec_center = tf.reduce_mean(pred_vec, axis=0, keepdims=True)
            img_center = tf.constant([(img.shape[2]-1) / 2, (img.shape[1]-1) / 2], tf.float32)
            inv_rot_pred_vec = tf.matmul(pred_vec - img_center, inv_rot_matrix) + img_center
            inv_rot_pred_bbox = create_bbox_from_vector(inv_rot_pred_vec)
            pred_vec = tf.matmul(inv_rot_pred_bbox - img_center, rot_matrix) + img_center
            if_pred = False
        else:
            if_pred = True

        # Prepare the vector or bbox for plotting
        vec_plot = prepare_vec_label2plot(vec, class_idx, pred=False)
        pred_vec_plot = prepare_vec_label2plot(pred_vec, pred_class_idx, pred=if_pred)

        # Plot the original image
        ax = axs[row, col+2]
        #ax.set_title(f'angle: {np.degrees(angle):.1f}', fontsize=8)
        ax.set_title(r'class: $\bf{x}$ pred:'.format(x=class_idx.numpy())+bold_argmax_print(ex_pred_class.numpy(),2), fontsize=7)
        ax.imshow(img[row])
        ax.scatter(x, y, marker='+', color='blue', s=80)
        ax.quiver(cx, cy, a_x, a_y, color='blue', width=0.02, scale=2, alpha=0.5)

        # Plot the vector or bbox
        if vec_plot is not None:
            ax.plot(*vec_plot, color='black', linewidth=5, alpha=0.5)
        if pred_vec_plot is not None:
            ax.plot(*pred_vec_plot, color='red', linewidth=2)

        ax.set_ylim(0.5, img_height-0.5)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()