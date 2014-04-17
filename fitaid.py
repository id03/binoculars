import sys
sys.path.append('/home/willem/Documents/PhD/python/BINoculars')

import os
import glob
from PyQt4 import QtGui, QtCore, Qt
from PyMca import QSpecFileWidget, QDataSource, StackBrowser, StackSelector
import BINoculars.main, BINoculars.space, BINoculars.plot
import numpy


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QTAgg
import matplotlib.figure, matplotlib.image

class Window(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        load_specfile = QtGui.QAction("Load specfile", self)  
        load_specfile.triggered.connect(self.load_specfile)

        load_configfile = QtGui.QAction("Load configfile", self)  
        load_configfile.triggered.connect(self.load_configfile)

        menu_bar = QtGui.QMenuBar() 
        file = menu_bar.addMenu("&File") 
        file.addAction(load_specfile) 
        file.addAction(load_configfile) 
      
        tab_widget = QtGui.QTabWidget() 
        self.fit_widget = FitWidget()

        tab_widget.addTab(self.fit_widget, "Fit CTR") 

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(menu_bar) 
        vbox.addWidget(tab_widget) 
        self.setLayout(vbox) 


    def load_specfile(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '.')
        self.mask_widget.spec_widget.setData(QDataSource.QDataSource(str(fname)))

    def load_configfile(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', '.', '*.txt'))
        self.config_widget.set_config_from_file(BINoculars.main.read_config_text(filename))
        
class FitWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(FitWidget, self).__init__(parent)

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        hbox = QtGui.QHBoxLayout() 
        left = QtGui.QVBoxLayout()
        right = QtGui.QVBoxLayout()

        right.addWidget(self.canvas)

        hbox.addLayout(left)
        hbox.addLayout(right)

        self.setLayout(hbox)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())





