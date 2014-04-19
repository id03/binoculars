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

        hbox = QtGui.QHBoxLayout() 

        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.make_leftpanel()

        splitter.addWidget(self.leftpanel)
        splitter.addWidget(self.canvas)

        hbox.addWidget(splitter) 

        self.setLayout(hbox)  


    def make_leftpanel(self):
        self.leftpanel = QtGui.QWidget()

        self.leftpanel.resolution_axis = QtGui.QComboBox()
        for ax in self.axes:
            self.leftpanel.resolution_axis.addItem(ax.label)
        self.leftpanel.resolution_line = QtGui.QLineEdit(str(self.axes[0].res))
        self.leftpanel.resolution_line.setMaximumWidth(50)
        self.leftpanel.resolution_button = QtGui.QPushButton('set')
        self.leftpanel.resolution_button.clicked.connect(self.set_res)

        self.leftpanel.navigation_button_left_end = QtGui.QPushButton('|<')
        self.leftpanel.navigation_button_left_one = QtGui.QPushButton('<')
        self.leftpanel.navigation_slider = QtGui.QSlider(Qt.Qt.Horizontal)
        self.leftpanel.navigation_slider.sliderReleased.connect(self.plot)

        self.leftpanel.navigation_button_right_one = QtGui.QPushButton('>')
        self.leftpanel.navigation_button_right_end = QtGui.QPushButton('>|')

        self.leftpanel.navigation_button_left_end.setMaximumWidth(20)
        self.leftpanel.navigation_button_left_one.setMaximumWidth(20)
        self.leftpanel.navigation_button_right_end.setMaximumWidth(20)
        self.leftpanel.navigation_button_right_one.setMaximumWidth(20)

        self.leftpanel.button_plot = QtGui.QPushButton('plot')
        self.leftpanel.button_plot.clicked.connect(self.plot)

        self.leftpanel.button_save = QtGui.QPushButton('save')
        self.leftpanel.button_save.clicked.connect(self.save)

        vbox = QtGui.QVBoxLayout() 

        vbox.addWidget(self.leftpanel.button_plot)
        vbox.addWidget(self.leftpanel.button_save)

        smallbox = QtGui.QHBoxLayout() 
        smallbox.addWidget(self.leftpanel.resolution_axis)
        smallbox.addWidget(self.leftpanel.resolution_line)
        smallbox.addWidget(self.leftpanel.resolution_button)

        lowbox = QtGui.QHBoxLayout() 
        lowbox.addWidget(self.leftpanel.navigation_button_left_end)
        lowbox.addWidget(self.leftpanel.navigation_button_left_one)
        lowbox.addWidget(self.leftpanel.navigation_slider)
        lowbox.addWidget(self.leftpanel.navigation_button_right_one)
        lowbox.addWidget(self.leftpanel.navigation_button_right_end)

        vbox.addLayout(smallbox)
        vbox.addLayout(lowbox)

        self.leftpanel.setLayout(vbox) 

    def plot(self):
        fitslice = self.slices[self.leftpanel.navigation_slider.value()]
        self.figure.clear()
        space = BINoculars.space.Space.fromfile(self.filename, fitslice.key)
        space = space.project(fitslice.axis)
        if fitslice.result is not None:
            if newspace.dimension == 1:
                self.ax = self.figure.add_subplot(111)
                BINoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit = fit)
            elif newspace.dimension == 2:
                pyplot.figure(figsize=(12, 9))
                pyplot.subplot(121)
                BINoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit = None)
                pyplot.subplot(122)
                BINoculars.plot.plot(newspace, pyplot.gcf(), pyplot.gca(), label=basename, log=not args.nolog, clipping=float(args.clip), fit = fit)
        else:
            self.ax = self.figure.add_subplot(111)
            BINoculars.plot.plot(space, self.figure, self.ax)  
        self.canvas.draw()

    def save(self):
        pass

    def set_res(self):
        self.slices = list()
        resolution = self.leftpanel.resolution_line.text()
        axindex = self.axes.index(str(self.leftpanel.resolution_axis.currentText()))
        ax = self.axes[axindex]
        axlabel = ax.label

        if float(resolution) < ax.res:
            raise ValueError('interval {0} to low, minimum interval is {1}'.format(resolution, ax.res))

        mi, ma = ax.min, ax.max
        bins = numpy.linspace(mi, ma, numpy.ceil(1 / numpy.float(resolution) * (ma - mi)) + 1)

        for start, stop in zip(bins[:-1], bins[1:]):
            key = [slice(None) for i in self.axes]
            key[axindex] = slice(start, stop)
            self.slices.append(FitSlice(key, axlabel))

        self.leftpanel.navigation_slider.setMinimum(0)
        self.leftpanel.navigation_slider.setMaximum(len(self.slices) - 1)
        self.leftpanel.navigation_slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.leftpanel.navigation_slider.setValue(0)

        self.leftpanel.navigation_button_left_end.clicked.connect(self.slider_change_left_end)
        self.leftpanel.navigation_button_left_one.clicked.connect(self.slider_change_left_one)
        self.leftpanel.navigation_button_right_end.clicked.connect(self.slider_change_right_end)
        self.leftpanel.navigation_button_right_one.clicked.connect(self.slider_change_right_one)

        self.plot()
        
    def slider_change_left_one(self):
        self.leftpanel.navigation_slider.setValue(max(self.leftpanel.navigation_slider.value() - 1, 0))
        self.plot()

    def slider_change_left_end(self):
        self.leftpanel.navigation_slider.setValue(0)
        self.plot()

    def slider_change_right_one(self):
        self.leftpanel.navigation_slider.setValue(min(self.leftpanel.navigation_slider.value() + 1, len(self.slices) - 1))
        self.plot()

    def slider_change_right_end(self):
        self.leftpanel.navigation_slider.setValue(len(self.slices) - 1)
        self.plot()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.resize(1000, 600)
    main.show()

    sys.exit(app.exec_())






