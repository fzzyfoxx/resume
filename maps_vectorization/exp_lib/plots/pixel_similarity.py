import tensorflow as tf
import argparse
from models_src.Attn_variations import SqueezeImg, UnSqueezeImg
from models_src.Metrics import WeightedF12D
from models_src.fft_lib import decode1Dcoords
from models_src.VecModels import PixelSimilarityF1
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--s", default=5, type=int, help="Number of columns/images input")
parser.add_argument("--next", default=1, type=int, help="Prepare plots for another batch")

kwargs = vars(parser.parse_args())

s = kwargs['s']
next_batch = kwargs['next']

if next_batch:
    img, labels = next(test_iter) # type: ignore
    labels = labels['Dot_Similarity']

ex_num = len(img)
dot_labels = tf.squeeze(SqueezeImg()(labels), axis=-1)

pattern_labels = tf.reduce_sum(dot_labels[:,1:] + tf.random.uniform((ex_num, dot_labels.shape[1]-1, 1), -0.5, 0.5), axis=1)
_, sample_points = tf.math.top_k(pattern_labels+tf.random.uniform(pattern_labels.shape, -0.1, 0.1), 1)
sample_points = sample_points[:,0]

dot_labels = tf.matmul(dot_labels, dot_labels, transpose_a=True)

preds = trainer.model(img, training=False)['Dot_Similarity'] # type: ignore
#print(labels.shape, preds.shape)
print('total F1:', WeightedF12D()(dot_labels, preds).numpy())

#sample_points = tf.random.uniform((s,), 0, 32**2, dtype=tf.int32)
sample_labels = UnSqueezeImg()(tf.gather(dot_labels, sample_points[...,tf.newaxis], axis=1, batch_dims=1)[:,0,:,tf.newaxis])
sample_preds = UnSqueezeImg()(tf.gather(preds, sample_points[...,tf.newaxis], axis=1, batch_dims=1)[:,0,:,tf.newaxis])

print('weighted F1:', PixelSimilarityF1()(labels, preds).numpy())
print('sample F1:', WeightedF12D()(sample_labels, sample_preds).numpy())

bin_sample_preds = tf.where(sample_preds>0.5, 1, 0)

sample_points = decode1Dcoords(sample_points, 32)

#print(sample_labels.shape, sample_preds.shape, sample_points.shape)

fig, axs = plt.subplots(4, s, figsize=(s*3, 4*3))

for i in range(s):
    axs[0,i].imshow(img[i]) 
    axs[1,i].imshow(sample_labels[i], vmin=0, vmax=1)
    axs[2,i].imshow(sample_preds[i], vmin=0, vmax=1)
    axs[3,i].imshow(bin_sample_preds[i], vmin=0, vmax=1)

    for j in range(4):
        axs[j,i].scatter(*sample_points[i], marker='+', color='red', s=100)

    for i, title in enumerate(['img', 'true_mask', 'pred_mask', 'pred_binarized']):
        axs[i,0].set_ylabel(title, rotation=90, fontsize=12)

plt.show()