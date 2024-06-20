import tensorflow as tf


class CombinedMetricsModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(CombinedMetricsModel, self).__init__(*args, **kwargs)

        self.custom_metrics = [tf.keras.metrics.Mean(name='loss')]
    
    def add_metrics(self, metrics):
        '''
            metrics: list of dicts with obligatory keys:
                - metric - function of metric
                - label - model output key
                - weight_label - label of y_true_matched to use to sample_weight
        '''
        for metric_def in metrics:
            metric = metric_def['metric']
            metric.label = metric_def['label']
            metric.weight_label = metric_def['weight_label']
            self.custom_metrics += [metric]

    '''def get_config(self):
        config = super().get_config()
        config.update({
            'custom_metrics': self.custom_metrics
        })
        return config'''

    @property
    def metrics(self):
        return self.custom_metrics
    
    def update_metrics(self, loss, y_true_matched, y_pred_matched):
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                weight_label = metric.weight_label
                sample_weight = y_true_matched[weight_label] if weight_label else None
                metric.update_state(y_true_matched[metric.label], y_pred_matched[metric.label], sample_weight=sample_weight)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss, y_true_matched, y_pred_matched = self.compute_loss(y=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        return self.update_metrics(loss, y_true_matched, y_pred_matched)
    
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss, y_true_matched, y_pred_matched = self.compute_loss(y=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)

        return self.update_metrics(loss, y_true_matched, y_pred_matched)

    def compute_loss(self, y=None, y_pred=None):
        return self.loss(y, y_pred)
    
    def save(self, *args, **kwargs):
        tf.keras.Model(self.input, self.output).save(*args, **kwargs)

