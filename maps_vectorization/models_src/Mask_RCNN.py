import tensorflow as tf
import transformers as t

def ResNet4Classification(input_shape=(256,256,3), FN_filters=[512], dropout=0.2, output_classes=4, config_args={}, name='ResNet', output_hidden_states=False):
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Permute((3,1,2))(inputs)
    x = t.TFResNetModel(config=t.ResNetConfig(**config_args))(x, output_hidden_states=output_hidden_states)
    x = tf.keras.layers.Flatten()(x['pooler_output'])
    x = tf.keras.layers.Dropout(dropout)(x)
    for filters in FN_filters:
        x = tf.keras.layers.Dense(filters, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name=name)

    return model