from PyQt5 import QtGui, QtCore, QtWidgets, uic
import sys
import cv2
import numpy as np

class AnotherWindow(QtWidgets.QWidget):
    def __init__(self,imglist,labellist):
        super().__init__()
        uic.loadUi('./labelandimg.ui', self)

        category = ['airplane','automobile','bird','cat','deer','dog','frog','horse','sheep','truck'] # label to name
        qtimglabellist = [self.label_1,self.label_2,self.label_3,self.label_4,self.label_5,self.label_6,self.label_7,self.label_8,self.label_9,self.label_10]
        qtlabellist = [self.label_11,self.label_12,self.label_13,self.label_14,self.label_15,self.label_16,self.label_17,self.label_18,self.label_19,self.label_20]

        for i in range(10):
            image = cv2.resize(imglist[i], dsize=(128, 128), interpolation = cv2.INTER_CUBIC)
            qtimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            qtimglabellist[i].setPixmap(QtGui.QPixmap.fromImage(qtimage))
            qtlabellist[i].setText(category[labellist[i]])
