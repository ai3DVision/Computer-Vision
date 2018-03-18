import tensorflow as tf
import numpy as np 
import os
from dataloader import Dataset
from model import Model
import argparse


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', dest='data_path', help='path of the data',
                        default='../data', type=str)
	parser.add_argument('--model_path', dest='model_path', help='path to store model',
                        default='./model', type=str)
	parser.add_argument('--batch_size', dest='batch_size', help='batch_size',
                        default='256', type=int)
	parser.add_argument('--epoch', dest='epoch', help='epoch',
                        default='30', type=int)
	parser.add_argument('--restore', dest='is_restore', help='restore or not',
                        default=False, type=bool)

	args = parser.parse_args()

	return args

args = parse_args() 


def train():
	idx = 0
	batch_size = args.batch_size
	learning_rate_start = 0.001
	epoch = args.epoch
	model_path = args.model_path
	data_path = args.data_path
	is_restore = args.is_restore

	data = Dataset('train', data_path)

	with tf.Graph().as_default():
		x, y, y_onehot, logits = Model(batch_size).build_model('train', keep_rate=0.5)
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits, name='softmax_loss'))
		reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		loss = tf.reduce_sum(cross_entropy) + tf.reduce_sum(reg_loss)
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, 10000, 0.1)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)
		tf.summary.scalar('loss', loss)
		summary = tf.summary.merge_all()

		with tf.Session() as sess:
			summary_writer = tf.summary.FileWriter('./graph', sess.graph)
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver(max_to_keep=5)
			if is_restore:
				try:
					saver.restore(sess, tf.train.latest_checkpoint(model_path))
				except:
					print('No model in ' + model_path + ' to restore')
					raise

			print('Start training...')
			while True:
				epoch_now, idx, imgs, labels = data.load_batch(batch_size, idx)
				loss_, _, step, summary_ = sess.run([loss, train_op, global_step, summary], feed_dict={x: imgs, y:labels})
				summary_writer.add_summary(summary_, global_step=step)
				if step%20 == 0:
					print('Epoch: {0}, Step: {1}, Loss: {2}' .format(epoch_now, step, loss_)) 
				if step%200 == 0:
					saver.save(sess, os.path.join(model_path, 'model.ckpt'), global_step=step)
				if epoch_now == epoch:
					print('Finish training ...')
					break


if __name__ == '__main__':
	print(args)
	train()
			 
