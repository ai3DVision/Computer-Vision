import tensorflow as tf

class Model:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_model(self, mode, keep_rate=1):
	
		x = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3], name='input_x')
		y = tf.placeholder(tf.int32, shape=[self.batch_size], name='input_y')
		y_onehot = tf.one_hot(y, depth=10)
		
		if mode == 'train':
			x_ = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), x)
		else:
			x_ = x
		
		conv1 = tf.layers.conv2d(x_, 64, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')

		bn1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
		conv2_1 = tf.layers.conv2d(bn1, 64, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2_1')
		conv2_2 = tf.nn.relu(tf.layers.batch_normalization(conv2_1))
		conv2_2 = tf.layers.conv2d(conv2_2, 64, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2_2')
		conv2_3 = tf.layers.conv2d(conv1, 64, (1, 1), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2_3')
		conv2_3 = tf.add(conv2_3, conv2_2)

		bn2 = tf.nn.relu(tf.layers.batch_normalization(conv2_3))
		conv3_1 = tf.layers.conv2d(bn2, 128, (3, 3), padding='same',strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3_1')
		conv3_2 = tf.nn.relu(tf.layers.batch_normalization(conv3_1))
		conv3_2 = tf.layers.conv2d(conv3_2, 128, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3_2')
		conv3_3 = tf.layers.conv2d(conv2_3, 128, (1, 1), strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3_3')
		conv3_3 = tf.add(conv3_3, conv3_2)
		
		bn3 = tf.nn.relu(tf.layers.batch_normalization(conv3_3))
		conv4_1 = tf.layers.conv2d(bn3, 256, (3, 3), padding='same', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4_1')
		conv4_2 = tf.nn.relu(tf.layers.batch_normalization(conv4_1))
		conv4_2 = tf.layers.conv2d(conv4_2, 256, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4_2')
		conv4_3 = tf.layers.conv2d(conv3_3, 256, (1, 1), strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4_3')
		conv4_3 = tf.add(conv4_3, conv4_2)
		conv4_3 = tf.nn.dropout(conv4_3, keep_rate)
		
		bn4 = tf.nn.relu(tf.layers.batch_normalization(conv4_3))
		conv5_1 = tf.layers.conv2d(bn4, 512, (3, 3), padding='same', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5_1')
		conv5_2 = tf.nn.relu(tf.layers.batch_normalization(conv5_1))
		conv5_2 = tf.layers.conv2d(conv5_2, 512, (3, 3), padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5_2')
		conv5_3 = tf.layers.conv2d(conv4_3, 512, (1, 1), strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv5_3')
		conv5_3 = tf.add(conv5_3, conv5_2)
		
		conv5_3_reshape = tf.nn.relu(tf.reshape(tf.nn.avg_pool(conv5_3, (1, 4, 4, 1), (1, 4, 4, 1), padding='VALID'), (self.batch_size, -1)))
		conv5_3_reshape = tf.nn.dropout(conv5_3_reshape, keep_rate)	
		logits = tf.layers.dense(conv5_3_reshape, 10, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4), name='fc')

		return x, y, y_onehot, logits
