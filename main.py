import sys
import cv2
import numpy as np
import pickle
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import random

#self define module
import loaddataset as ld
import loadimg


class MainUi(QtWidgets.QMainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		uic.loadUi('./main.ui', self)
		self.iniGuiEvent()

	def iniGuiEvent(self):# connect all button to all event slot
		self.pushButton_showImg.clicked.connect(self.pushButton_showImg_onClick)


	#5.1 show data image
	@QtCore.pyqtSlot()
	def pushButton_showImg_onClick(self):
		batch1 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_1')
		batch2 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_2')
		batch3 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_3')
		batch4 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_4')
		batch5 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_5')

		imglist = []
		labellist = []
		for i in range(10):
			index = random.randint(0,9999)
			img = np.transpose(np.reshape(batch1[b'data'][index],(3, 32,32)), (1,2,0))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #convert color channel
			imglist.append(img)
			label = batch1[b'labels'][index]
			labellist.append(label)

		self.w = loadimg.AnotherWindow(imglist,labellist)
		self.w.show()




if __name__ == "__main__": #main function
	def run_app():
		app = QtWidgets.QApplication(sys.argv)
		window = MainUi()
		window.show()
		app.exec_()
	run_app()