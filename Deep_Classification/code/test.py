import tensorflow as tf
import argparse
from dataloader import Dataset
from model import Model
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', dest='data_path', help='path of the data',
                        default='../data', type=str)
	parser.add_argument('--model_path', dest='model_path', help='path to restore model',
                        default='./model', type=str)
	parser.add_argument('--batch_size', dest='batch_size', help='batch_size',
                        default='16', type=int)
	parser.add_argument('--data_mode', dest='data_mode', help='training data or testing data',
                        default='test', type=str)

	args = parser.parse_args()
	
	return args

args = parse_args()


def test():
	epoch = 1
	batch_size = args.batch_size
	model_path = args.model_path
	data_path = args.data_path
	idx = 0
	confusion_matrix_total = np.zeros((10, 10))
	data_mode = args.data_mode	

	data = Dataset(data_mode, data_path, is_shuffle=False)

	with tf.Graph().as_default():
		x, y, y_onehot, logits = Model(batch_size).build_model('test', keep_rate=1)
		prediction = tf.argmax(logits, axis=1)
		accuracy, accuracy_update = tf.metrics.accuracy(labels=y, predictions=prediction)
		batch_confusion = tf.confusion_matrix(labels=y, predictions=prediction, num_classes=10)

		with tf.Session() as sess:
			sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
			restorer = tf.train.Saver()

			try:
				restorer.restore(sess, tf.train.latest_checkpoint(model_path))
			except:
				print('No model in ' + model_path + ' to restore')
				raise

			while True:
				epoch_now, idx, imgs, labels = data.load_batch(batch_size, idx)
				_, batch_confusion_ = sess.run([accuracy_update, batch_confusion], feed_dict={x: imgs, y:labels})
				confusion_matrix_total += batch_confusion_ 
				if epoch_now == epoch:
					np.save('./confusion.npy', confusion_matrix_total)
					break

			accuracy_ = sess.run(accuracy)
			return accuracy_

if __name__ == '__main__':
	print(args)
	accuracy = test()
	print('Testing accuracy: {0}%' .format(accuracy*100))

