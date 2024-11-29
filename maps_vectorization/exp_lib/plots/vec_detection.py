import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from models_src.VecModels import prepare_vec_label2plot
from IPython.display import clear_output
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--s", default=5, type=int, help="Number of columns/images input")
parser.add_argument("--r", default=8, type=int, help="Number of rows/sample points")

kwargs = vars(parser.parse_args())

s = kwargs['s']
max_rows = kwargs['r']

def gather_sorted_components(x, components_class):
    return tf.concat([tf.gather(x, tf.where(components_class==2)[...,0], axis=0),
                                 tf.gather(x, tf.where(components_class==1)[...,0], axis=0)], axis=0)

def bold_argmax_print(x, round_digits=2):
    i_max = np.argmax(x)
    output = [str(round(elem,round_digits)) if i!=i_max else r'$\bf{x}$'.format(x=str(round(elem,round_digits))) for i,elem in enumerate(x)]
    output = ', '.join(output)

    return output

features, labels, labels_weights = next(test_iter) # type: ignore
img, sample_points = features.values()
vecs_labels, components_class, thickness_label = labels.values()
vecs_weights, class_weights = tf.where(labels_weights['vecs']>0, 1.0, 0.0), labels_weights['class']

pred_vecs, pred_class, pred_thickness = trainer.model(features, training=False).values() # type: ignore
clear_output()

print('img', img.shape, img.dtype)
print('sample_points', sample_points.shape, sample_points.dtype)
print('vec_labels', vecs_labels.shape, vecs_labels.dtype)
print('components_class', components_class.shape, components_class.dtype)
print('vecs_weights', vecs_weights.shape, vecs_weights.dtype)
print('class_weights', class_weights.shape, class_weights.dtype)


components_class = tf.argmax(components_class, axis=-1)
components_class_pred = tf.argmax(pred_class, axis=-1)
#pred_class = tf.argmax(pred_class, axis=-1)

print('pred_vecs', pred_vecs.shape, pred_vecs.dtype)
print('pred_class', pred_class.shape, pred_class.dtype)


label_colors = tf.one_hot(components_class, 3)
pred_colors = tf.one_hot(components_class_pred, 3)
stacked_colors = tf.cast(tf.stack([label_colors, pred_colors], axis=-3)*255, tf.int32)


i=0

rows = min(max_rows, cfg.sample_points) # type: ignore

print('\n background: RED, bbox: GREEN, line: BLUE \n')

fig, axs = plt.subplots(rows+1, s, figsize=(s*3, (rows+1)*3)) 

for i in range(s):
    ex_vec_labels = vecs_labels[i]
    ex_vec_pred = pred_vecs[i]
    ex_class_idxs = components_class[i]
    ex_pred_class = pred_class[i]
    ex_sample_points = sample_points[i]
    ex_pred_class_idxs = components_class_pred[i]

    axs[0,i].imshow(stacked_colors[i])
    axs[0,i].set_yticks([0,1],['label', 'pred'], fontsize=8)
    axs[0,i].set_xticks(np.arange(0, cfg.sample_points, 2), np.arange(0, cfg.sample_points, 2), fontsize=6) # type: ignore

    for r in range(rows): #range(cfg.sample_points):
        sample_point = ex_sample_points[r]
        vec_label = ex_vec_labels[r]
        vec_pred = ex_vec_pred[r]
        class_idx = ex_class_idxs[r]
        pred_class_idx = ex_pred_class_idxs[r]

        vec_label = prepare_vec_label2plot(vec_label, class_idx, pred=False)
        vec_pred = prepare_vec_label2plot(vec_pred, pred_class_idx, pred=True)
        ax = axs[r+1,i]
        #r = r if r%2==0 else r//2+cfg.sample_points//2
        ax.set_title(r'class: $\bf{x}$ pred:'.format(x=class_idx.numpy())+bold_argmax_print(ex_pred_class[r].numpy(),2), fontsize=8)
        ax.imshow(img[i])
        ax.scatter(*sample_point[::-1], marker='+', color='blue', s=150)
        if vec_label is not None:
            ax.plot(*vec_label, color='black', linewidth=6, alpha=0.8)
        if vec_pred is not None:
            ax.plot(*vec_pred, color='red', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()