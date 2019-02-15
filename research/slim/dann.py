import tensorflow as tf
from tensorflow.python.framework import ops

slim = tf.contrib.slim

######################
# Define the network #
######################
def revgrad(features, num_domain_layers, num_classes, domain_adaptation_weight, weight_decay, batch_size, train=True):
    with tf.variable_scope(
            'main', 
            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay), 
        ) as sc:

        features = tf.expand_dims(tf.expand_dims(features, 1), 1)
        features = slim.conv2d(features, 1024, [1,1], activation_fn=tf.nn.relu,
                          normalizer_fn=None, scope='layer1')
        features = slim.conv2d(features, 1024, [1,1], activation_fn=tf.nn.relu,
                          normalizer_fn=None, scope='layer2')

        classify_features = features[:batch_size,:] if train else features

        class_logits = slim.conv2d(classify_features, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
        class_logits = tf.squeeze(class_logits, [1, 2], name='SpatialSqueeze')

        with tf.variable_scope('domain_adaptation'):
            @ops.RegisterGradient('GradientReversal')
            def _flip_gradients(op, grad):
                return [tf.negative(grad) * domain_adaptation_weight]
        
            g = tf.get_default_graph()
            with g.gradient_override_map({"Identity": 'GradientReversal'}):
                domain_features = tf.identity(features, name='gradient_reversal')

            for i in range(num_domain_layers):
                domain_features = slim.conv2d(domain_features, 1024, [1,1], activation_fn=tf.nn.relu,
                                       normalizer_fn=None, scope='layer{}'.format(i+1))
            domain_logits = slim.conv2d(domain_features, 2, [1,1], activation_fn=None,
                                   normalizer_fn=None, scope='logits')
            domain_logits = tf.squeeze(domain_logits, [1, 2], name='SpatialSqueeze')

    return class_logits, domain_logits


def no_revgrad(features, num_classes, weight_decay):
    with tf.variable_scope(
            'main', 
            regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay), 
        ) as sc:

        features = tf.expand_dims(tf.expand_dims(features, 1), 1)
        features = slim.conv2d(features, 1024, [1,1], activation_fn=tf.nn.relu,
                          normalizer_fn=None, scope='layer1')
        features = slim.conv2d(features, 1024, [1,1], activation_fn=tf.nn.relu,
                          normalizer_fn=None, scope='layer2')

        class_logits = slim.conv2d(features, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='logits')
        class_logits = tf.squeeze(class_logits, [1, 2], name='SpatialSqueeze')

    return class_logits
