import numpy as np
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

import keras
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D,MaxPooling2D,LeakyReLU
from keras.models import Model,load_model
from keras.layers import Lambda,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import callbacks
from sklearn import metrics
from sklearn import utils


class Img_Classifier(object):
	"""docstring for Img_Classifier"""
	def __init__(self):

		self.img_rows = 32
		self.img_cols = 32
		self.img_ch = 3
		self.input_shape = (self.img_rows,self.img_cols,self.img_ch)
		self.num_classes = 2

		self.batch_size = 50

		self.train_dir = 'data/idrid_aug_sampled/train/'
		self.valid_dir = 'data/idrid_aug_sampled/valid/'
		self.test_dir = 'data/idrid_aug_sampled/test/'

		self.seed = seed
		self.name = 'ResNet50_random_init'
		self.save_path = 'models/'+self.name+'_best.h5'
		self.log_path = 'logs/'+self.name+'.txt'
		self.train_log_path = 'logs/train_log_'+self.name+'.csv'

		self.cws = None

		print(self.name)

	def get_model(self):
		return self.get_pt_model()

	def get_pt_model(self):
		img_inp = Input(shape=self.input_shape)
		base_model = ResNet50(include_top=False, 
			# weights='imagenet',
			weights=None,
			input_tensor = img_inp, input_shape=self.input_shape, pooling='avg')
		x = base_model.output
		pred = Dense(self.num_classes,activation='sigmoid')(x)
		model = Model(img_inp,pred,name='img_clf')
		model.summary()
		return model

	def get_train_gen(self):

		img_gen = ImageDataGenerator(
			# rotation_range=0.1,
			# width_shift_range=0.1,
			# height_shift_range=0.1,
			# shear_range=0.1,
			# zoom_range=0.1,
			# fill_mode='wrap',
			# cval=0.,
			horizontal_flip=True,
			vertical_flip=True,
			rescale=1/255.0,
			)

		img_gen = img_gen.flow_from_directory(
			self.train_dir,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb',
			shuffle = True,
			)

		y = img_gen.classes
		# self.cws = utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
		# print(self.cws)
		return img_gen


	def get_valid_gen(self):
		img_gen = ImageDataGenerator(
			rescale=1/255.0,
			)

		img_gen = img_gen.flow_from_directory(
			self.valid_dir,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb',
			shuffle = True,
			)
		return img_gen


	def get_test_gen(self):
		img_gen = ImageDataGenerator(
			rescale=1/255.0,
			)

		img_gen = img_gen.flow_from_directory(
			self.test_dir,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb',
			shuffle = False,
			)

		return img_gen

	def get_callbacks(self):
		checkpointer = callbacks.ModelCheckpoint(filepath=self.save_path, monitor='val_loss', verbose=1, save_best_only=True)
		early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=10, verbose=1, mode='auto')
		reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
		csv_logger = callbacks.CSVLogger(self.train_log_path, separator=',', append=False)
		# tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		return [early_stopping,checkpointer,reduce_lr,csv_logger]

	def build_model(self,lr):
		model = self.get_model()
		opt = Adam(lr,beta_1=.9,beta_2=.999)
		model.compile(
			loss = 'binary_crossentropy',
			optimizer = opt,
			metrics = ['accuracy']
			)
		return model

	def train(self,lr,epochs):

		model =self.build_model(lr)
		train_gen = self.get_train_gen()
		valid_gen = self.get_valid_gen()

		model.fit_generator(
			generator = train_gen,
			epochs = epochs,
			validation_data = valid_gen,
			callbacks = self.get_callbacks(),
			# class_weight=self.cws,
			# steps_per_epoch=10,
			# validation_steps=10,
			)

	def get_pred(self,use_saved=False):

		if not use_saved :
			model = load_model(self.save_path)
			test_gen = self.get_test_gen()
			y_pred_prob = model.predict_generator(
				test_gen, verbose = 1,
				# steps=10,
				)
			y_true = test_gen.classes
			np.savez_compressed('tmp/'+self.name+'.npz',y_pred_prob=y_pred_prob,y_true=y_true)
			return y_true,y_pred_prob
		else:
			data = np.load('tmp/'+self.name+'.npz')
			return data['y_true'],data['y_pred_prob']


	def calc_aucs(self,y_true,y_pred_prob,i):
		fpr, tpr, _ = metrics.roc_curve(y_true[:, i], y_pred_prob[:, i])
		prec, rec, _ = metrics.precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
		# prg_curve = prg.create_prg_curve(y_true[:, i], y_pred_prob[:, i])
		auroc = metrics.auc(fpr, tpr)
		auprc = metrics.auc(prec, rec, reorder=True)
		# auprg = prg.calc_auprg(prg_curve)
		return auroc, auprc

	def evaluate(self,save_report=True,use_saved=False):

		y_true,y_pred_prob = self.get_pred(use_saved=use_saved)
		y_true = to_categorical(y_true,num_classes=self.num_classes)
		print(y_pred_prob.shape)

		log_file = open(self.log_path,'w+')

		for i in range(self.num_classes):
			auroc, auprc = self.calc_aucs(y_true,y_pred_prob,i=i)
			print('Class %d auroc %.3f auprc %.3f'%(i,auroc,auprc))
			print('Class %d auroc %.3f auprc %.3f'%(i,auroc,auprc),file=log_file)

		y_pred = np.argmax(y_pred_prob,axis=1)
		y_true = np.argmax(y_true,axis=1)
		f1 = metrics.f1_score(y_true,y_pred,average='weighted')
		acc = metrics.accuracy_score(y_true,y_pred)
		cm = metrics.confusion_matrix(y_true,y_pred)
		print('F1 score %.3f  Acc : %.3f'%(f1,acc))
		print('F1 score %.3f  Acc : %.3f'%(f1,acc),file=log_file)
		print(cm)
		print(cm,file=log_file)
		print(seed)
		log_file.close()


	def debug(self):

		for gen in [self.get_train_gen(),self.get_valid_gen(),self.get_test_gen()]:
			print('*****************')
			for i in range(2):
				x,y = next(gen)
				print(x.shape)
				print(y.shape)
				print(y)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu')
	parser.add_argument('--seed')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	seed = int(args.seed)
	np.random.seed(seed)
	tf.set_random_seed(seed)

	clf = Img_Classifier()
	clf.debug()
	clf.train(lr=1e-4,epochs=100)
	clf.evaluate(save_report=True,use_saved=False)