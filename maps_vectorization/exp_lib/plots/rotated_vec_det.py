import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from models_src.VecModels import prepare_vec_label2plot
from models_src.VecModels import NoSplitMixedBboxVecMultiPropMetric, gen_rot_matrix_yx
from IPython.display import clear_output
from exp_lib.plots.plot_utils import bold_argmax_print
import argparse
import random
import cv2
from models_src.VecModels import prepare_vec_label2plot
from models_src.VecModels import NoSplitMixedBboxVecMultiPropMetric, gen_rot_matrix_yx

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
fig, axs = plt.subplots(r, num_examples_per_image, figsize=(num_examples_per_image * 2, r * 2))

# Iterate over the number of rows (images)
for row in range(r):
    # Iterate over the number of examples per image
    for col in range(num_examples_per_image):
        # Select random points where vec_weights > 0
        valid_points = tf.where(vec_weights[row] > 0)
        selected_point = random.choice(valid_points.numpy())
        y, x = selected_point[0], selected_point[1]
        
        # Get angle with the lowest loss
        #angle_idx = np.argmin(proposals_vec_loss[row, y, x])
        angle_idx = np.argmax(vec_pred[row, y, x, :, -1])
        angle = angle_input[row, angle_idx]
        

        # Extract the vector label for the selected point
        vec = vec_label[row, y, x, angle_idx]
        class_idx = tf.argmax(shape_class[row, y, x])
        ex_pred_class = class_pred[row, y, x, angle_idx]   
        pred_class_idx = tf.argmax(ex_pred_class)

        pred_vec = tf.reshape(vec_pred[row, y, x, angle_idx, :4], (2, 2))

        rot_matrix = gen_rot_matrix_yx(angle)
        inv_rot_matrix = gen_rot_matrix_yx(-angle)

        img_center = tf.constant([(img.shape[2]-1) / 2, (img.shape[1]-1) / 2], tf.float32)
        rot_vec_label = tf.matmul(vec - img_center, inv_rot_matrix) + img_center
        rot_pred_vec = tf.matmul(pred_vec - img_center, inv_rot_matrix) + img_center

        rot_selected_point = (tf.matmul(tf.constant([[y, x]], tf.float32) - img_center, inv_rot_matrix) + img_center)[0]
        rot_y, rot_x = rot_selected_point[0], rot_selected_point[1]

        # Prepare the vector or bbox for plotting
        vec_plot = prepare_vec_label2plot(rot_vec_label, class_idx, pred=False)
        pred_vec_plot = prepare_vec_label2plot(rot_pred_vec, pred_class_idx, pred=True)

        # Rotate the image using OpenCV
        (h, w) = img.shape[1:3]
        M = cv2.getRotationMatrix2D(img_center.numpy(), np.degrees(angle), 1.0)
        rotated_img_cv = cv2.warpAffine(img[row].numpy(), M, (w, h))

        # Plot the original image
        ax = axs[row, col]
        #ax.set_title(f'angle: {np.degrees(angle):.1f}', fontsize=8)
        ax.set_title(r'class: $\bf{x}$ pred:'.format(x=class_idx.numpy())+bold_argmax_print(ex_pred_class.numpy(),2), fontsize=7)
        ax.imshow(rotated_img_cv)
        ax.scatter(rot_x, rot_y, marker='+', color='blue', s=80)

        # Plot the vector or bbox
        if vec_plot is not None:
            ax.plot(*vec_plot, color='black', linewidth=4, alpha=0.5)
        if pred_vec_plot is not None:
            ax.plot(*pred_vec_plot, color='red', linewidth=2)

        ax.set_ylim(0.5, img_height-0.5)
        ax.set_xticks([])
        ax.set_yticks([])