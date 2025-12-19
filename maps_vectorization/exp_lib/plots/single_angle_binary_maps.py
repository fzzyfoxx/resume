import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--r", default=8, type=int, help="Number of rows/image inputs")
parser.add_argument("--next", default=1, type=int, help="Prepare plots for another batch")

kwargs = vars(parser.parse_args())

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
max_thickness = max(np.max(thickness_label), np.max(thickness_pred))

# Create a figure for plotting
fig, axs = plt.subplots(r, 9, figsize=(10 * 2, r * 2))

# Iterate over the number of rows (images)
for row in range(r):
    plot_imgs = [img[row], shape_class[row,...,0], class_pred[row,...,0], shape_class[row,...,1], class_pred[row,...,1], shape_class[row,...,2], class_pred[row,...,2], \
                 thickness_label[row], thickness_pred[row]*tf.where(thickness_weights[row,...,tf.newaxis]>0, 1., 0.)]
    plot_labels = ['img', 'background', 'background_pred', 'bbox', 'bbox_pred', 'line', 'line_pred', 'thickness', 'thickness_pred']

    for i, (plot_img, title) in enumerate(zip(plot_imgs, plot_labels)):
        ax = axs[row, i]
        if 'thickness' in title:
            ax.imshow(plot_img, vmin=0, vmax=max_thickness)
        else:
            ax.imshow(plot_img)
        ax.set_title(title, fontsize=7)
        ax.set_ylim(0.5, img_height-0.5)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()