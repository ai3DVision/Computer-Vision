import pickle
import numpy as np
from random import shuffle
import os 

mean = np.array([0.485, 0.456, 0.406])
std= np.array([0.229, 0.224, 0.225])

class Dataset:
	def __init__(self, mode, data_path, is_shuffle=True):
		print('Load cifar10 ' + mode + ' data ...')
		if mode == 'train':
			with open(os.path.join(data_path, 'cifar10_train.pkl'), 'rb') as f:
				self.data = pickle.load(f)
		elif mode == 'test':
			with open(os.path.join(data_path, 'cifar10_test.pkl'), 'rb') as f:
				self.data = pickle.load(f)
		print('Load cifar10 ' + mode + ' data Success !')
		self.total_length = len(self.data['labels'])
		self.idx_list = list(range(self.total_length))
		if is_shuffle:
			shuffle(self.idx_list)
		self.epoch = 0

	def load_batch(self, batch_size, idx):
		imgs = []
		labels = []
		if idx + batch_size >= self.total_length:
			cut_point = idx + batch_size - self.total_length + 1
			idx_list = self.idx_list[idx:-1] + self.idx_list[:cut_point]
			idx = cut_point
			self.epoch += 1
		else:
			idx_list = self.idx_list[idx:idx+batch_size]

		for data_idx in idx_list:
			img = self.data['images'][data_idx]
			label = self.data['labels'][data_idx]
			img = (img.astype('float32')/255 - mean)/std
			imgs.append(img)
			labels.append(label)
		idx = idx + batch_size
		imgs = np.asarray(imgs)
		labels = np.asarray(labels)

		return self.epoch, idx, imgs, labels
