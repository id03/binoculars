import sys
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

        load_hdf5file = QtGui.QAction("Load mesh", self)  
        load_hdf5file.triggered.connect(self.load_hdf5file)

        menu_bar = QtGui.QMenuBar() 
        file = menu_bar.addMenu("&File") 
        file.addAction(load_hdf5file) 

        self.tab_widget = QtGui.QTabWidget() 

                 
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(menu_bar) 
        self.vbox.addWidget(self.tab_widget) 
        self.setLayout(self.vbox) 

    def load_hdf5file(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', '.', '*.hdf5'))

        self.plot_widget = FitWidget(filename)
        self.tab_widget.addTab(self.plot_widget, '{0}'.format(filename.split('/')[-1]))
        self.setLayout(self.vbox)


class FitSlice(object):
    def __init__(self, key, axis):
        self.key = key
        self.axis = axis
        self.result = None
        
                
class FitWidget(QtGui.QWidget):
    def __init__(self, filename ,parent=None):
        super(FitWidget, self).__init__(parent)

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.filename = filename
        self.axes = BINoculars.space.Axes.fromfile(filename)

        self.slices = list()

        self.resolution_axis = QtGui.QComboBox()
        for ax in self.axes:
            self.resolution_axis.addItem(ax.label)
        self.resolution_line = QtGui.QLineEdit()
        self.resolution_line.setMaximumWidth(50)
        self.resolution_button = QtGui.QPushButton('set')
        self.resolution_button.clicked.connect(self.set_res)

        self.navigation_button_left_end = QtGui.QPushButton('|<')
        self.navigation_button_left_one = QtGui.QPushButton('<')
        self.navigation_slider = QtGui.QSlider(Qt.Qt.Horizontal)
        self.navigation_button_right_end = QtGui.QPushButton('>')
        self.navigation_button_right_one = QtGui.QPushButton('>|')

        self.navigation_button_left_end.setMaximumWidth(20)
        self.navigation_button_left_one.setMaximumWidth(20)
        self.navigation_button_right_end.setMaximumWidth(20)
        self.navigation_button_right_one.setMaximumWidth(20)

        hbox = QtGui.QHBoxLayout() 
        left = QtGui.QVBoxLayout()
        right = QtGui.QVBoxLayout()

        self.button_plot = QtGui.QPushButton('plot')
        self.button_plot.clicked.connect(self.plot)

        self.button_save = QtGui.QPushButton('save')
        self.button_save.clicked.connect(self.save)

        left.addWidget(self.button_plot)
        left.addWidget(self.button_save)

        smallbox = QtGui.QHBoxLayout() 
        smallbox.addWidget(self.resolution_axis)
        smallbox.addWidget(self.resolution_line)
        smallbox.addWidget(self.resolution_button)

        lowbox = QtGui.QHBoxLayout() 
        lowbox.addWidget(self.navigation_button_left_end)
        lowbox.addWidget(self.navigation_button_left_one)
        lowbox.addWidget(self.navigation_slider)
        lowbox.addWidget(self.navigation_button_right_end)
        lowbox.addWidget(self.navigation_button_right_one)

        left.addLayout(smallbox)
        left.addLayout(lowbox)       
        right.addWidget(self.canvas)

        hbox.addLayout(left)
        hbox.addLayout(right)

        self.setLayout(hbox)  

    def plot(self):
        pass

    def save(self):
        pass

    def set_res(self):
        resolution = self.resolution_line.text()
        axindex = self.axes.index(str(self.resolution_axis.currentText()))
        ax = self.axes[axindex]
        axlabel = ax.label

        if float(resolution) < ax.res:
            raise ValueError('interval {0} to low, minimum interval is {1}'.format(resolution, ax.res))

        mi, ma = ax.min, ax.max
        bins = numpy.linspace(mi, ma, numpy.ceil(1 / numpy.float(resolution) * (ma - mi)) + 1)

        for start, stop in zip(bins[:-1], bins[1:]):
            key = [slice(None) for i in self.axes]
            key[axindex] = slice(start, stop)
            self.slices.append(FitSlice(key,axlabel))

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.resize(1000, 600)
    main.show()

    sys.exit(app.exec_())






