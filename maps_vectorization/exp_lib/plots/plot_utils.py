import tensorflow as tf
import numpy as np

def create_bbox_from_vector(x):
    p1, p3 = x[0], x[1]
    # Calculate the other two points of the bbox
    p2 = tf.stack([p3[0], p1[1]])
    p4 = tf.stack([p1[0], p3[1]])
    return tf.stack([p1, p3, p2, p4], axis=0)

def bold_argmax_print(x, round_digits=2):
    i_max = np.argmax(x)
    output = [str(round(elem,round_digits)) if i!=i_max else r'$\bf{x}$'.format(x=str(round(elem,round_digits))) for i,elem in enumerate(x)]
    output = ', '.join(output)

    return output