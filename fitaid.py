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

    def load_hdf5file(self, filename = None):
        if filename is None:
            filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', '.', '*.hdf5'))
        self.plot_widget = FitWidget(filename)
        self.tab_widget.addTab(self.plot_widget, '{0}'.format(filename.split('/')[-1]))
        self.setLayout(self.vbox)


class FitSlice(object):
    def __init__(self, key, axis):
        self.key = key
        self.axis = axis
        self.result = None
        


class ButtonedSlider(QtGui.QWidget):
    def __init__(self,parent=None):
        super(ButtonedSlider, self).__init__(parent)

        self.navigation_button_left_end = QtGui.QPushButton('|<')
        self.navigation_button_left_one = QtGui.QPushButton('<')
        self.navigation_slider = QtGui.QSlider(Qt.Qt.Horizontal)
        self.navigation_slider.sliderReleased.connect(self.send)

        self.navigation_button_right_one = QtGui.QPushButton('>')
        self.navigation_button_right_end = QtGui.QPushButton('>|')

        self.navigation_button_left_end.setMaximumWidth(20)
        self.navigation_button_left_one.setMaximumWidth(20)
        self.navigation_button_right_end.setMaximumWidth(20)
        self.navigation_button_right_one.setMaximumWidth(20)

        self.navigation_button_left_end.clicked.connect(self.slider_change_left_end)
        self.navigation_button_left_one.clicked.connect(self.slider_change_left_one)
        self.navigation_button_right_end.clicked.connect(self.slider_change_right_end)
        self.navigation_button_right_one.clicked.connect(self.slider_change_right_one)

        box = QtGui.QHBoxLayout() 
        box.addWidget(self.navigation_button_left_end)
        box.addWidget(self.navigation_button_left_one)
        box.addWidget(self.navigation_slider)
        box.addWidget(self.navigation_button_right_one)
        box.addWidget(self.navigation_button_right_end)

        self.setDisabled(True)
        self.setLayout(box)

    def set_length(self,length):
        self.navigation_slider.setMinimum(0)
        self.navigation_slider.setMaximum(length - 1)
        self.navigation_slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.navigation_slider.setValue(0)
        self.setEnabled(True)


    def send(self):
        self.emit(QtCore.SIGNAL('slice_index'), self.navigation_slider.value())

    def slider_change_left_one(self):
        self.navigation_slider.setValue(max(self.navigation_slider.value() - 1, 0))
        self.send()

    def slider_change_left_end(self):
        self.navigation_slider.setValue(0)
        self.send()

    def slider_change_right_one(self):
        self.navigation_slider.setValue(min(self.navigation_slider.value() + 1, self.navigation_slider.maximum()))
        self.send()

    def slider_change_right_end(self):
        self.navigation_slider.setValue(self.navigation_slider.maximum())
        self.send()

    def index(self):
        return self.navigation_slider.value()


class ControlWidget(QtGui.QWidget):
    def __init__(self, axes ,parent=None):
        super(ControlWidget, self).__init__(parent)

        self.axes = axes

        self.resolution_axis = QtGui.QComboBox()
        for ax in self.axes:
            self.resolution_axis.addItem(ax.label)

        self.resolution_line = QtGui.QLineEdit(str(self.axes[0].res))
        self.resolution_line.setMaximumWidth(50)
        self.resolution_button = QtGui.QPushButton('set')
        self.resolution_button.clicked.connect(parent.set_res)

        self.button_plot = QtGui.QPushButton('plot')
        self.button_plot.clicked.connect(parent.plot)

        self.button_save = QtGui.QPushButton('save')
        self.button_save.clicked.connect(parent.save)

        
        self.nav = ButtonedSlider()
        self.nav.connect(self.nav, QtCore.SIGNAL('slice_index'), parent.plot)

        vbox = QtGui.QVBoxLayout() 

        vbox.addWidget(self.button_plot)
        vbox.addWidget(self.button_save)

        smallbox = QtGui.QHBoxLayout() 
        smallbox.addWidget(self.resolution_axis)
        smallbox.addWidget(self.resolution_line)
        smallbox.addWidget(self.resolution_button)

        vbox.addLayout(smallbox)
        vbox.addWidget(self.nav)
        self.setLayout(vbox) 

    @property
    def res(self):
        return self.resolution_line.text()

    @property
    def axis(self):
        return str(self.resolution_axis.currentText())

    @property
    def index(self):
        return str(self.resolution_axis.currentText())

    def set_length(self, length):
        self.nav.set_length(length)
                
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
        self.control_widget = ControlWidget(self.axes, self)

        splitter.addWidget(self.control_widget)
        splitter.addWidget(self.canvas)

        hbox.addWidget(splitter) 

        self.setLayout(hbox)  

    def plot(self, index = None):
        if index == None:
            fitslice = self.slices[self.control_widget.index]
        else:
            fitslice = self.slices[index]

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

    def set_res(self, axis = None, resolution = None):

        self.slices = list()

        if not resolution:
            resolution = self.control_widget.res
        if not axis:
            axis = self.control_widget.axis

        axindex = self.axes.index(axis)
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

        self.control_widget.set_length(len(self.slices))
        self.plot()
        

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.resize(1000, 600)
    #main.load_hdf5file('mesh_5781-5788.hdf5')
    #main.plot_widget.set_res(axis = 'l')
    main.show()

    
    sys.exit(app.exec_())






