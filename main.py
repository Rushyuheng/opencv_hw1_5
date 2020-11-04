import sys
import cv2
import numpy as np
import pickle
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#self define module
import loaddataset as ld
import loadimg


class MainUi(QtWidgets.QMainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		uic.loadUi('./main.ui', self)
		self.iniGuiEvent()
		self.setupmodel()

	def setupmodel(self):
		self.model = models.Sequential()
		self.model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
		self.model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.MaxPooling2D((2, 2)))

		self.model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.MaxPooling2D((2, 2)))

		self.model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.MaxPooling2D((2, 2)))

		self.model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.MaxPooling2D((2, 2)))

		self.model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
		self.model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))

		self.model.add(layers.Flatten())  # 2*2*512
		self.model.add(layers.Dense(4096, activation='relu'))
		self.model.add(layers.Dropout(0.5))
		self.model.add(layers.Dense(4096, activation='relu'))
		self.model.add(layers.Dropout(0.5))
		self.model.add(layers.Dense(10, activation='softmax'))

	def iniGuiEvent(self):# connect all button to all event slot
		self.pushButton_showImg.clicked.connect(self.pushButton_showImg_onClick)
		self.pushButton_ShowPara.clicked.connect(self.pushButton_ShowPara_onClick)
		self.pushButton_ShowStruct.clicked.connect(self.pushButton_ShowStruct_onClick)


	#5.1 show data image
	@QtCore.pyqtSlot()
	def pushButton_showImg_onClick(self):
		batch1 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_1')
		batch2 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_2')
		batch3 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_3')
		batch4 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_4')
		batch5 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_5')

		batchlist = [batch1,batch2,batch3,batch4,batch5]
		imglist = []
		labellist = []
		for i in range(10):
			index = random.randint(0,9999)
			batchindex = random.randint(0,4)
			img = np.transpose(np.reshape(batchlist[batchindex][b'data'][index],(3, 32,32)), (1,2,0))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #convert color channel
			imglist.append(img)
			label = batchlist[batchindex][b'labels'][index]
			labellist.append(label)

		self.w = loadimg.AnotherWindow(imglist,labellist)
		self.w.show()

	#5.2 show hyperparameter structure
	@QtCore.pyqtSlot()
	def pushButton_ShowPara_onClick(self):
		batchsize = 100
		learningrate = 0.001
		adma = tf.keras.optimizers.Adam(learning_rate=learningrate)
		self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
		print('hyperparameter:')
		print('batchsize: %d' %batchsize)
		print('learning rate: %.3f' %learningrate)
		print('optimizer: Adam')


	#5.3 show model structure
	@QtCore.pyqtSlot()
	def pushButton_ShowStruct_onClick(self):
		self.model.summary()


if __name__ == "__main__": #main function
	def run_app():
		app = QtWidgets.QApplication(sys.argv)
		window = MainUi()
		window.show()
		app.exec_()
	run_app()