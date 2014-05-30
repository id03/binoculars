import sys
import os
from PyQt4 import QtGui, QtCore, Qt
import BINoculars.main, BINoculars.space, BINoculars.plot, BINoculars.fit
import numpy
import inspect


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QTAgg
import matplotlib.figure, matplotlib.image

class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        load_hdf5file = QtGui.QAction("Load mesh", self)  
        load_hdf5file.triggered.connect(self.load_hdf5file)

        menu_bar = QtGui.QMenuBar() 
        file = menu_bar.addMenu("&File") 
        file.addAction(load_hdf5file) 

        self.tab_widget = QtGui.QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.tab_widget.removeTab)

        self.setMenuBar(menu_bar)
        self.statusbar	= QtGui.QStatusBar()

        self.setCentralWidget(self.tab_widget)
        self.setMenuBar(menu_bar)
        self.setStatusBar(self.statusbar)

    def load_hdf5file(self, filename = False):
        if not filename:
            filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', '.', '*.hdf5'))
        self.plot_widget = FitWidget(filename)
        self.tab_widget.addTab(self.plot_widget, '{0}'.format(filename.split('/')[-1]))
       
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

        self.parent = parent
        self.axes = axes


        self.resolution_axis = QtGui.QComboBox()
        for ax in self.axes:
            self.resolution_axis.addItem(ax.label)
        QtCore.QObject.connect(self.resolution_axis, QtCore.SIGNAL("currentIndexChanged(int)"), parent.set_res)

        self.functions = list()
        self.function_box = QtGui.QComboBox()        
        for function in dir(BINoculars.fit):
            cls = getattr(BINoculars.fit, function)
            if isinstance(cls, type) and issubclass(cls, BINoculars.fit.PeakFitBase):
                self.functions.append(cls)
                self.function_box.addItem(function)

        self.parambox = QtGui.QComboBox()
        for param in inspect.getargspec(self.fitclass.func).args[1]:
            self.parambox.addItem(param)


        self.log = QtGui.QCheckBox('log')
        self.log.setChecked(True)
        QtCore.QObject.connect(self.log, QtCore.SIGNAL("stateChanged(int)"), parent.log_changed)


        QtCore.QObject.connect(self.function_box, QtCore.SIGNAL("activated(int)"), self.fitfunctionchanged)
        QtCore.QObject.connect(self.parambox, QtCore.SIGNAL("activated(int)"), parent.plot_overview)

        self.resolution_line = QtGui.QLineEdit(str(self.axes[0].res))
        self.resolution_line.setMaximumWidth(50)
        QtCore.QObject.connect(self.resolution_line, QtCore.SIGNAL("editingFinished()"), parent.set_res)

        self.button_save = QtGui.QPushButton('save')
        self.button_save.clicked.connect(parent.save)

        self.fit_all = QtGui.QPushButton('fit all')
        self.fit_all.clicked.connect(parent.fit_all)

        self.fit = QtGui.QPushButton('fit')
        self.fit.clicked.connect(parent.fit)

        self.nav = ButtonedSlider()
        self.nav.connect(self.nav, QtCore.SIGNAL('slice_index'), parent.index_callback)

        self.succesbox = QtGui.QCheckBox('fit succesful')
        self.succesbox.setChecked(True)
        QtCore.QObject.connect(self.succesbox, QtCore.SIGNAL("stateChanged(int)"), parent.fitsucces_changed)

        self.button_show = QtGui.QPushButton('show')
        self.button_show.clicked.connect(parent.log_changed)


        vbox = QtGui.QVBoxLayout() 
        vbox.addWidget(self.button_save)
        vbox.addWidget(self.function_box)
        vbox.addWidget(self.fit_all)
        vbox.addWidget(self.fit)

        smallbox = QtGui.QHBoxLayout() 
        smallbox.addWidget(self.resolution_axis)
        smallbox.addWidget(self.resolution_line)

        overviewbox = QtGui.QHBoxLayout()
        overviewbox.addWidget(self.button_show) 
        overviewbox.addWidget(self.parambox)
        overviewbox.addWidget(self.log)

        vbox.addLayout(smallbox)
        vbox.addWidget(self.nav)
        vbox.addWidget(self.succesbox)        
        vbox.addLayout(overviewbox)
        self.setLayout(vbox) 

    def fitfunctionchanged(self, index):
        self.parent.fitfunction_callback()
        self.parambox.clear()
        for param in inspect.getargspec(self.fitclass.func).args[1]:
            self.parambox.addItem(param)

    @property
    def res(self):
        return self.resolution_line.text()

    @property
    def axis(self):
        return str(self.resolution_axis.currentText())

    @property
    def index(self):
        return int(self.nav.index())

    @property
    def fitclass(self):
        return self.functions[self.function_box.currentIndex()]

    def set_length(self, length):
        self.nav.set_length(length)


class HiddenToolbar(NavigationToolbar2QTAgg):
    def __init__(self, corner_callback, canvas):
        NavigationToolbar2QTAgg.__init__(self, canvas, None)
        self._corner_callback = corner_callback
        self.zoom()

    def press(self, event):
        self._corner_preclick = self._views()
        
    def release(self, event):
        if self._corner_preclick == self._views():
            self._corner_callback(event.xdata, event.ydata)
        self._corner_preclick = None
                
class FitWidget(QtGui.QWidget):
    def __init__(self, filename ,parent=None):
        super(FitWidget, self).__init__(parent)

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = HiddenToolbar(self.loc_callback, self.canvas)

        self.roddict = dict()

        self.filename = filename
        self.axes = BINoculars.space.Axes.fromfile(filename)

        hbox = QtGui.QHBoxLayout() 

        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.control_widget = ControlWidget(self.axes, self)

        splitter.addWidget(self.control_widget)
        splitter.addWidget(self.canvas)

        hbox.addWidget(splitter) 
        self.set_res()
        self.setLayout(hbox)  

    def plot(self):
        self.figure.clear()
        space = self.rod.get_space()
        fit = self.rod.get_slice().fit
        self.control_widget.succesbox.setChecked(self.rod.get_slice().succes)

        if fit is not None:
            if space.dimension == 1:
                self.ax = self.figure.add_subplot(111)
                BINoculars.plot.plot(space, self.figure, self.ax, fit = fit.fitdata)
            elif space.dimension == 2:
                self.ax = self.figure.add_subplot(121)
                BINoculars.plot.plot(space, self.figure, self.ax, fit = None)
                self.ax = self.figure.add_subplot(122)
                BINoculars.plot.plot(space, self.figure, self.ax, fit = fit.fitdata)
        else:
            self.ax = self.figure.add_subplot(111)
            BINoculars.plot.plot(space, self.figure, self.ax)  
        self.canvas.draw()

    def plot_overview(self, index):
        x = list()
        y = list()
        for fitslice in self.rod.slices:
            if fitslice.succes:
                x.append(fitslice.coord)
                y.append(fitslice.fit.result[index])

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(x,y,'+')
        if self.control_widget.log.checkState():
            self.ax.semilogy()
        self.canvas.draw()

    def log_changed(self, log):
        self.plot_overview(self.control_widget.parambox.currentIndex())

    def save(self):
        dialog = QtGui.QFileDialog(self, "Save image");
        dialog.setFilter('Portable Network Graphics (*.png);;Portable Document Format (*.pdf)');
        dialog.setDefaultSuffix('png');
        dialog.setFileMode(QtGui.QFileDialog.AnyFile);
        dialog.setAcceptMode(QtGui.QFileDialog.AcceptSave);
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        try:
            self.figure.savefig(str(fname))
        except Exception as e:
            QtGui.QMessageBox.critical(self, 'Save image', 'Unable to save image to {}: {}'.format(fname, e))
    
    def set_res(self, axis = None, resolution = None):
        if not resolution:
            resolution = self.control_widget.res
        if not axis:
            axis = self.control_widget.axis

        key = (str(resolution), self.axes.index(axis))
        if key in self.roddict:
            self.rod = self.roddict[key]
        else:
            self.rod = FitRod(self.filename, axis, resolution)
            self.roddict[key] = self.rod

        self.control_widget.set_length(len(self.rod.slices))
        self.fitfunction_callback()
        self.plot()
                    
    def loc_callback(self, x, y):
        if self.ax:
            fitslice = self.rod.get_slice()
            fitslice.loc = numpy.array([x, y])
            self.fit()

    def index_callback(self, index):
        self.rod.set_index(index)
        self.plot()

    def fitfunction_callback(self, index = None):
        self.rod.set_function(self.control_widget.fitclass)

    def fitsucces_changed(self, state):
        fitslice = self.rod.get_slice()
        fitslice.succes = bool(state)

    def fit(self):
        if self.rod:
            self.rod.fit()
            self.plot()

    def fit_all(self):
        if self.rod:
            self.rod.fit_all()
            self.plot()

class FitRod(object):
    def __init__(self, filename, axis, resolution):
        self.slices = list()
        self.filename = filename
        self.axis = axis
        self.axes = BINoculars.space.Axes.fromfile(filename)
        self.index = 0

        axindex = self.axes.index(axis)
        ax = self.axes[axindex]
        axlabel = ax.label

        if float(resolution) < ax.res:
            raise ValueError('interval {0} to low, minimum interval is {1}'.format(resolution, ax.res))

        mi, ma = ax.min, ax.max
        bins = numpy.linspace(mi, ma, numpy.ceil(1 / numpy.float(resolution) * (ma - mi)) + 1)

        for start, stop in zip(bins[:-1], bins[1:]):
            k = [slice(None) for i in self.axes]
            k[axindex] = slice(start, stop)
            coord = (start + stop) / 2
            self.slices.append(FitSlice(k, coord))

    def get_space(self):
        return BINoculars.space.Space.fromfile(self.filename, self.slices[self.index].key).project(self.axis)

    def get_slice(self):
        return self.slices[self.index]

    def set_index(self,index):
        self.index = index

    def set_function(self, function):
        self.function = function

    def fit_all(self):
        pd = QtGui.QProgressDialog('Performing fit...', 'Cancel', 0, len(self.slices))
        pd.setWindowModality(QtCore.Qt.WindowModal)
        pd.show()
        def progress(i):
            pd.setValue(i)
            if pd.wasCanceled():
                raise KeyboardInterrupt
            QtGui.QApplication.processEvents()
            return self.fit(i)
        for index in range(len(self.slices)):
             progress(index)
        pd.close()

    def fit(self, index = None):
        if not index:
            fitslice = self.get_slice()
        else:
            fitslice = self.slices[index]

        space = BINoculars.space.Space.fromfile(self.filename, fitslice.key).project(self.axis)
        fitslice.fit = self.function(space, loc = fitslice.loc)
        fitslice.succes = True

class FitSlice(object):
    def __init__(self, key, coord):
        self.key = key
        self.coord = coord
        self.fit = None
        self.loc = None
        self.succes = False

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.resize(1000, 600)
    main.load_hdf5file('rod_217.hdf5')
    #main.plot_widget.set_res(axis = 'l')
    main.show()

    
    sys.exit(app.exec_())






