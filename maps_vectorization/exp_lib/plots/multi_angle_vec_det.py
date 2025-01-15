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
parser.add_argument("--angle_type", default=1, type=int, help="Angle prediction gathering type. 0 for loss argmin, 1 for max prob")

kwargs = vars(parser.parse_args())

num_examples_per_image = kwargs['s']
r = kwargs['r']
next_batch = kwargs['next']
max_prob = bool(kwargs['angle_type'])

if max_prob:
    print('Max probability prediction for angle mode')
else:
    print('Loss argmin prediction for angle mode')


if next_batch:
    print('Processing next batch')
    inputs, labels, weights = next(test_iter) # type: ignore
    img, angle_input = inputs['img'], inputs['angle_input']
    vec_label, shape_class, thickness_label = labels['vecs'], labels['class'], labels['thickness']
    vec_weights, class_weights, thickness_weights = weights['vecs'], weights['class'], weights['thickness']

    preds = trainer.model(inputs, training=False) # type: ignore
    vec_pred, class_pred, thickness_pred = preds['vecs'], preds['class'], preds['thickness']

# Create a metric for the vector loss
vec_metric = NoSplitMixedBboxVecMultiPropMetric(size=32, gamma=1, norm=False)

img_height = img.shape[1]

y_true, y_pred = vec_metric.format_inputs(vec_label, vec_pred)
proposals_vec_loss = tf.reduce_min(vec_metric._vec_loss(y_true, y_pred), axis=-1)
print('proposals_vec_loss', proposals_vec_loss.shape)

# Create a figure for plotting
fig, axs = plt.subplots(r, num_examples_per_image+1, figsize=((num_examples_per_image+1) * 2, r * 2))

# Iterate over the number of rows (images)
for row in range(r):
    # Iterate over the number of examples per image
    a_y = tf.sin(angle_input[row])
    a_x = tf.cos(angle_input[row])
    angles_num = angle_input.shape[-1]

    cx, cy = [np.array([x]*angles_num) for x in [(img.shape[2]-1) / 2, (img.shape[1]-1) / 2]]

    ax = axs[row, 0]
    ax.imshow(img[row])
    ax.quiver(cx, cy, a_x, a_y, color='black', width=0.02, scale=2, alpha=0.75)
    ax.set_title(f'img {row} angle inputs', fontsize=7)
    ax.set_ylim(0.5, img_height-0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    for col in range(num_examples_per_image):
        # Select random points where vec_weights > 0
        valid_points = tf.where(vec_weights[row] > 0)
        selected_point = random.choice(valid_points.numpy())
        y, x = selected_point[0], selected_point[1]
        
        # Get angle with the lowest loss
        if max_prob:
            angle_idx = np.argmax(vec_pred[row, y, x, :, -1])
        else:
            angle_idx = np.argmin(proposals_vec_loss[row, y, x])
        angle = angle_input[row, angle_idx]
        
        a_y = tf.sin(angle)
        a_x = tf.cos(angle)
        cx, cy = (img.shape[2]-1) / 2, (img.shape[1]-1) / 2

        # Extract the vector label for the selected point
        vec = vec_label[row, y, x, angle_idx]
        class_idx = tf.argmax(shape_class[row, y, x])
        ex_pred_class = class_pred[row, y, x, angle_idx]   
        pred_class_idx = tf.argmax(ex_pred_class)

        pred_vec = tf.reshape(vec_pred[row, y, x, angle_idx, :4], (2, 2))
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
        ax = axs[row, col+1]
        #ax.set_title(f'angle: {np.degrees(angle):.1f}', fontsize=8)
        ax.set_title(r'class: $\bf{x}$ pred:'.format(x=class_idx.numpy())+bold_argmax_print(ex_pred_class.numpy(),2), fontsize=7)
        ax.imshow(img[row])
        ax.scatter(x, y, marker='+', color='blue', s=80)
        ax.quiver(cx, cy, a_x, a_y, color='blue', width=0.02, scale=2, alpha=0.5)

        # Plot the vector or bbox
        if vec_plot is not None:
            ax.plot(*vec_plot, color='black', linewidth=4, alpha=0.5)
        if pred_vec_plot is not None:
            ax.plot(*pred_vec_plot, color='red', linewidth=2)

        ax.set_ylim(0.5, img_height-0.5)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()