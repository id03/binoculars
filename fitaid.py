import sys
import os
from PyQt4 import QtGui, QtCore, Qt
import BINoculars.main, BINoculars.space, BINoculars.plot, BINoculars.fit
import numpy
import json
import inspect
from scipy.interpolate import griddata

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QTAgg
import matplotlib.figure, matplotlib.image
from matplotlib.pyplot import Rectangle


def interpolate(space):
    data = space.get_masked()
    mask = data.mask
    grid = numpy.vstack(numpy.ma.array(g, mask=mask).compressed() for g in space.get_grid()).T
    open = numpy.vstack(numpy.ma.array(g, mask=numpy.invert(mask)).compressed() for g in space.get_grid()).T
    if open.shape[0] == 0:
        return data.compressed()
    elif grid.shape[0] == 0:
        return data.compressed()
    else:
        interpolated = griddata(grid, data.compressed(), open)            
        values = data.data.copy()
        values[mask] = interpolated
        mask = numpy.isnan(values)
        if mask.sum() > 0:
            data = numpy.ma.array(values, mask = mask)
            grid = numpy.vstack(numpy.ma.array(g, mask=mask).compressed() for g in space.get_grid()).T
            open = numpy.vstack(numpy.ma.array(g, mask=numpy.invert(mask)).compressed() for g in space.get_grid()).T
            interpolated = griddata(grid, data.compressed(), open, method = 'nearest')
            values[mask] = interpolated
        return values

class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        newproject = QtGui.QAction("New project", self)  
        newproject.triggered.connect(self.newproject)

        loadproject = QtGui.QAction("Open project", self)  
        loadproject.triggered.connect(self.loadproject)

        saveproject = QtGui.QAction("Save project", self)  
        saveproject.triggered.connect(self.saveproject)

        export = QtGui.QAction("Export fitdata", self)  
        export.triggered.connect(self.export)

        menu_bar = QtGui.QMenuBar() 
        file = menu_bar.addMenu("&File") 
        file.addAction(newproject)
        file.addAction(loadproject)
        file.addAction(saveproject) 

        edit = menu_bar.addMenu("&Edit") 
        edit.addAction(export)

        self.tab_widget = QtGui.QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        QtCore.QObject.connect(self.tab_widget, QtCore.SIGNAL("tabCloseRequested(int)"), self.tab_widget.removeTab)


        self.setMenuBar(menu_bar)
        self.statusbar	= QtGui.QStatusBar()

        self.setCentralWidget(self.tab_widget)
        self.setMenuBar(menu_bar)
        self.setStatusBar(self.statusbar)

    def newproject(self, filename = None):
        if not filename:
            dialog = QtGui.QFileDialog(self, "Load space");
            dialog.setFilter('BINoculars space file (*.hdf5)');
            dialog.setFileMode(QtGui.QFileDialog.ExistingFiles);
            dialog.setAcceptMode(QtGui.QFileDialog.AcceptOpen);
            if not dialog.exec_():
                return
            fname = dialog.selectedFiles()
            if not fname:
                return
            for name in fname:
                #try:
                widget = FitWidget(str(name), parent = self)
                self.tab_widget.addTab(widget, short_filename(str(name)))
                #except Exception as e:
                #    QtGui.QMessageBox.critical(self, 'Load space', 'Unable to load space from {}: {}'.format(fname, e))
        else:
            widget = FitWidget(filename, parent = self)
            self.tab_widget.addTab(widget, short_filename(filename))
            
    def loadproject(self, filename = None):
        if not filename:
            dialog = QtGui.QFileDialog(self, "Load project");
            dialog.setFilter('BINoculars fit file (*.fit)');
            dialog.setFileMode(QtGui.QFileDialog.ExistingFiles);
            dialog.setAcceptMode(QtGui.QFileDialog.AcceptOpen);
            if not dialog.exec_():
                return
            fname = dialog.selectedFiles()
            if not fname:
                return
            for name in fname:
                #try:
                widget = FitWidget.fromfile(str(name), parent = self)
                self.tab_widget.addTab(widget, short_filename(str(name)))
                #except Exception as e:
                #    QtGui.QMessageBox.critical(self, 'Load project', 'Unable to load project from {}: {}'.format(fname, e))
        else:
            widget = FitWidget.fromfile(filename, parent = self)
            self.tab_widget.addTab(widget, short_filename(filename))

    def saveproject(self):
        widget = self.tab_widget.currentWidget()
        dialog = QtGui.QFileDialog(self, "Save project");
        dialog.setFilter('BINoculars fit file (*.fit)');
        dialog.setDefaultSuffix('fit');
        dialog.setFileMode(QtGui.QFileDialog.AnyFile);
        dialog.setAcceptMode(QtGui.QFileDialog.AcceptSave);
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        #try:
        index = self.tab_widget.currentIndex()
        self.tab_widget.setTabText(index, short_filename(fname))
        widget.tofile(fname)
        #except Exception as e:
        #    QtGui.QMessageBox.critical(self, 'Save project', 'Unable to save project to {}: {}'.format(fname, e))

       
    def export(self):
        widget = self.tab_widget.currentWidget()
        dialog = QtGui.QFileDialog(self, "export fitdata");
        dialog.setFilter('text (*.txt)');
        dialog.setDefaultSuffix('txt');
        dialog.setFileMode(QtGui.QFileDialog.AnyFile);
        dialog.setAcceptMode(QtGui.QFileDialog.AcceptSave);
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        try:
            index = self.tab_widget.currentIndex()
            widget.export(fname)
        except Exception as e:
            QtGui.QMessageBox.critical(self, 'export fitdata', 'Unable to export fitdata to {}: {}'.format(fname, e))

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

class IntegrateWidget(QtGui.QGroupBox):
    def __init__(self,title , axes,parent=None):
        super(IntegrateWidget, self).__init__(title,parent)

        self.axes = axes

        integratebox = QtGui.QVBoxLayout()
        intensitybox = QtGui.QHBoxLayout()
        backgroundbox = QtGui.QHBoxLayout()

        self.hsize = QtGui.QDoubleSpinBox()
        self.vsize = QtGui.QDoubleSpinBox()

        QtCore.QObject.connect(self.hsize, QtCore.SIGNAL("valueChanged(double)"), self.send)
        QtCore.QObject.connect(self.vsize, QtCore.SIGNAL("valueChanged(double)"), self.send)

        intensitybox.addWidget(QtGui.QLabel('roi size:'))
        intensitybox.addWidget(self.hsize)
        intensitybox.addWidget(self.vsize)

        self.left = QtGui.QDoubleSpinBox()
        self.right = QtGui.QDoubleSpinBox()
        self.top = QtGui.QDoubleSpinBox()
        self.bottom = QtGui.QDoubleSpinBox()

        QtCore.QObject.connect(self.left, QtCore.SIGNAL("valueChanged(double)"), self.send)
        QtCore.QObject.connect(self.right, QtCore.SIGNAL("valueChanged(double)"), self.send)
        QtCore.QObject.connect(self.top, QtCore.SIGNAL("valueChanged(double)"), self.send)
        QtCore.QObject.connect(self.bottom, QtCore.SIGNAL("valueChanged(double)"), self.send)

        backgroundbox.addWidget(QtGui.QLabel('background'))
        backgroundbox.addWidget(self.left)
        backgroundbox.addWidget(self.right)
        backgroundbox.addWidget(self.top)
        backgroundbox.addWidget(self.bottom)
        
        self.integratebutton = QtGui.QPushButton('integrate')
        self.integratebutton.clicked.connect(self.integrate)
        
        integratebox.addLayout(intensitybox)
        integratebox.addLayout(backgroundbox)
        integratebox.addWidget(self.integratebutton)        
        self.setLayout(integratebox)

    def set_axis(self, axis):
        indices = range(self.axes.dimension)
        indices.pop(self.axes.index(axis))        
        
        self.newaxes = BINoculars.space.Axes(self.axes[index] for index in indices)

        self.hsize.setSingleStep(self.newaxes[1].res)
        self.hsize.setDecimals(len(str(self.newaxes[1].res)) - 2)
        self.vsize.setSingleStep(self.newaxes[0].res)
        self.vsize.setDecimals(len(str(self.newaxes[0].res)) - 2)
        self.left.setSingleStep(self.newaxes[1].res)
        self.left.setDecimals(len(str(self.newaxes[1].res)) - 2)
        self.right.setSingleStep(self.newaxes[1].res)
        self.right.setDecimals(len(str(self.newaxes[1].res)) - 2)
        self.top.setSingleStep(self.newaxes[0].res)
        self.top.setDecimals(len(str(self.newaxes[0].res)) - 2)
        self.bottom.setSingleStep(self.newaxes[0].res)
        self.bottom.setDecimals(len(str(self.newaxes[0].res)) - 2)

    def send(self):
        self.emit(QtCore.SIGNAL("valueChanged"))

    def integrate(self):
        self.emit(QtCore.SIGNAL("integrate"))
    
    def intkey(self, coords):
        vsize = self.vsize.value() / 2
        hsize = self.hsize.value() / 2
        return tuple(ax.restrict(slice(coord - size, coord + size)) for ax, coord, size in zip(self.newaxes, coords, [vsize, hsize]))

    def bkgkeys(self, coords):
        key = self.intkey(coords)

        vsize = self.vsize.value() / 2
        hsize = self.hsize.value() / 2

        leftkey = (key[0], self.newaxes[1].restrict(slice(coords[1] - hsize - self.left.value(), coords[1] - hsize)))
        rightkey = (key[0], self.newaxes[1].restrict(slice(coords[1] + hsize, coords[1] + hsize + self.right.value())))
        topkey = (self.newaxes[0].restrict(slice(coords[0] - vsize - self.top.value(), coords[0] - vsize)), key[1])
        bottomkey =  (self.newaxes[0].restrict(slice(coords[0] + vsize, coords[0] + vsize  + self.bottom.value())), key[1])

        return leftkey, rightkey, topkey, bottomkey

    def tolist(self):
        return [self.hsize.value(), self.vsize.value(), self.left.value() ,self.right.value() ,self.top.value(),   self.bottom.value()]

    def values_from_list(self, values):
        for box, value in zip([self.hsize, self.vsize, self.left, self.right, self.top, self.bottom], values):
            box.setValue(value)

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
        self.make_controlwidget()

        splitter.addWidget(self.control_widget)
        splitter.addWidget(self.canvas)

        hbox.addWidget(splitter) 
        self.set_res()
        self.setLayout(hbox)  

    def make_controlwidget(self):
        self.control_widget = QtGui.QWidget()

        self.resolution_axis = QtGui.QComboBox()
        for ax in self.axes:
            self.resolution_axis.addItem(ax.label)
        QtCore.QObject.connect(self.resolution_axis, QtCore.SIGNAL("currentIndexChanged(int)"), self.set_res)

        self.functions = list()
        self.function_box = QtGui.QComboBox()        
        for function in dir(BINoculars.fit):
            cls = getattr(BINoculars.fit, function)
            if isinstance(cls, type) and issubclass(cls, BINoculars.fit.PeakFitBase):
                self.functions.append(cls)
                self.function_box.addItem(function)

        self.paramlabel = QtGui.QLabel('projection axis:')
        self.parambox = QtGui.QComboBox()
        self.resolutionlabel = QtGui.QLabel('slice size of slice:')

        for param in inspect.getargspec(self.fitclass.func).args[1]:
            self.parambox.addItem(param)

        self.log = QtGui.QCheckBox('log')
        self.log.setChecked(True)
        QtCore.QObject.connect(self.log, QtCore.SIGNAL("stateChanged(int)"), self.log_changed)
        QtCore.QObject.connect(self.function_box, QtCore.SIGNAL("activated(int)"), self.fitfunctionchanged)
        QtCore.QObject.connect(self.parambox, QtCore.SIGNAL("activated(int)"), self.plot_overview)

        self.resolution_line = QtGui.QLineEdit(str(self.axes[0].res))
        self.resolution_line.setMaximumWidth(50)
        QtCore.QObject.connect(self.resolution_line, QtCore.SIGNAL("editingFinished()"), self.set_res)

        self.button_save = QtGui.QPushButton('save')
        self.button_save.clicked.connect(self.save)

        self.fit_all_button = QtGui.QPushButton('fit all')
        self.fit_all_button.clicked.connect(self.fit_all)

        self.fit_button = QtGui.QPushButton('fit')
        self.fit_button.clicked.connect(self.fit)

        self.fitting = QtGui.QGroupBox('Fitting')
        flayout = QtGui.QHBoxLayout()
        flayout.addWidget(self.function_box)
        flayout.addWidget(self.fit_all_button)
        flayout.addWidget(self.fit_button)
        self.fitting.setLayout(flayout)

        self.nav = ButtonedSlider()
        self.nav.connect(self.nav, QtCore.SIGNAL('slice_index'), self.index_callback)

        self.succesbox = QtGui.QCheckBox('fit succesful')
        self.succesbox.setChecked(True)
        QtCore.QObject.connect(self.succesbox, QtCore.SIGNAL("stateChanged(int)"), self.fitsucces_changed)

        self.button_show = QtGui.QPushButton('show')
        self.button_show.clicked.connect(self.log_changed)

        self.intwidget = IntegrateWidget('Integration', self.axes)
        QtCore.QObject.connect(self.intwidget, QtCore.SIGNAL("valueChanged"), self.plot_box)
        QtCore.QObject.connect(self.intwidget, QtCore.SIGNAL("integrate"), self.integrate)

        vbox = QtGui.QVBoxLayout() 
        vbox.addWidget(self.button_save)

        smallbox = QtGui.QHBoxLayout()
        smallbox.addWidget(self.paramlabel) 
        smallbox.addWidget(self.resolution_axis)
        smallbox.addWidget(self.resolutionlabel) 
        smallbox.addWidget(self.resolution_line)

        vbox.addLayout(smallbox)
        vbox.addWidget(self.fitting)
        vbox.addWidget(self.intwidget)

        overviewbox = QtGui.QHBoxLayout()
        overviewbox.addWidget(self.button_show) 
        overviewbox.addWidget(self.parambox)
        overviewbox.addWidget(self.log)

        vbox.addWidget(self.nav)
        vbox.addWidget(self.succesbox)        
        vbox.addLayout(overviewbox)
        self.control_widget.setLayout(vbox) 

    def fitfunctionchanged(self, index):
        self.fitfunction_callback()
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

    def plot(self):
        self.figure.clear()
        space = self.rod.get_space()
        self.figure.space_axes = space.axes
        fitslice = self.rod.get_slice()

        if hasattr(fitslice, 'fitdata'):
            fitdata = fitslice.fitdata
        elif hasattr(fitslice, 'result'):
            xdata, ydata, cxdata, cydata = self.rod.function._prepare(space)
            fitdata = numpy.ma.array(self.rod.function.func(xdata, fitslice.result), mask=ydata.mask)
        else:
            fitdata = None

        self.succesbox.setChecked(self.rod.get_slice().succes)

        if fitdata is not None:
            if space.dimension == 1:
                self.ax = self.figure.add_subplot(111)
                BINoculars.plot.plot(space, self.figure, self.ax, fit = fitdata)
            elif space.dimension == 2:
                self.ax = self.figure.add_subplot(121)
                BINoculars.plot.plot(space, self.figure, self.ax, fit = None)
                self.ax = self.figure.add_subplot(122)
                BINoculars.plot.plot(space, self.figure, self.ax, fit = fitdata)
        else:
            self.ax = self.figure.add_subplot(111)
            BINoculars.plot.plot(space, self.figure, self.ax)

        self.canvas.draw()
        self.plot_box()

    def plot_box(self):
        ax = self.figure.get_axes()[0]
        axes = self.figure.space_axes
        fitslice = self.rod.get_slice()
        if fitslice.loc != None:
            key = self.intwidget.intkey(fitslice.loc)
            bkgkey = self.intwidget.bkgkeys(fitslice.loc)

            ax.patches = []
            rect = Rectangle((key[0].start, key[1].start), key[0].stop - key[0].start, key[1].stop - key[1].start, alpha = 0.2,color =  'k')
            ax.add_patch(rect)
            for k in bkgkey:
                bkg = Rectangle((k[0].start, k[1].start), k[0].stop - k[0].start, k[1].stop - k[1].start, alpha = 0.2,color =  'r')
                ax.add_patch(bkg)
        self.canvas.draw()
          
    def plot_overview(self, index):
        curves = []
        curves.append(numpy.vstack(numpy.array([fitslice.coord, fitslice.result[index]]) for fitslice in self.rod.succesful()))
        param = str(self.parambox.currentText())
        if param.startswith('loc'):
            self.fit_loc()
            curves.append(numpy.vstack(numpy.array([fitslice.coord, fitslice.loc[int(param.split('loc')[-1])] ]) for fitslice in self.rod.slices))
        elif param.startswith('I'):
            if hasattr(self.rod.slices[0], 'intensity'):
                curves.append(numpy.vstack(numpy.array([fitslice.coord,fitslice.intensity]) for fitslice in self.rod.slices))
            if hasattr(self.rod.slices[0], 'sum'):
                curves.append(numpy.vstack(numpy.array([fitslice.coord,fitslice.sum]) for fitslice in self.rod.slices))

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        for curve in curves:
            self.ax.plot(curve[:,0], curve[:,1], '+')
        if self.log.checkState():
            self.ax.semilogy()
        self.canvas.draw()

    def fit_loc(self):
        indices = []
        for index, param in enumerate(inspect.getargspec(self.fitclass.func).args[1]):
            if param.startswith('loc'):
                indices.append(index)
        self.rod.fit_loc(indices)

    def log_changed(self, log):
        self.plot_overview(self.parambox.currentIndex())

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
            resolution = self.res
        if not axis:
            axis = self.axis

        key = (str(resolution), self.axes.index(axis))
        if key in self.roddict:
            self.rod = self.roddict[key]
        else:
            self.rod = FitRod(self.filename, axis, resolution)
            self.roddict[key] = self.rod

        self.set_length(len(self.rod.slices))
        self.intwidget.set_axis(axis)
        self.fitfunction_callback()
        self.plot()
                    
    def loc_callback(self, x, y):
        if self.ax:
            fitslice = self.rod.get_slice()
            fitslice.loc = numpy.array([x, y])
            self.fit()
            self.plot_box()

    def index_callback(self, index):
        self.rod.set_index(index)
        self.plot()

    def fitfunction_callback(self, index = None):
        self.rod.set_function(self.fitclass)

    def fitsucces_changed(self, state):
        fitslice = self.rod.get_slice()
        fitslice.succes = bool(state)

    def integrate(self):
        pd = QtGui.QProgressDialog('Integrating...', 'Cancel', 0, len(self.rod.slices))
        pd.setWindowModality(QtCore.Qt.WindowModal)
        pd.show()
        def progress(i, fitslice):
            pd.setValue(i)
            if pd.wasCanceled():
                raise KeyboardInterrupt
            QtGui.QApplication.processEvents()
            space = self.rod.get_space(index)
            key = self.intwidget.intkey(fitslice.loc)
            bkgkey = self.intwidget.bkgkeys(fitslice.loc)
            fitslice.intensity = self.rod.integrate(fitslice, space, key, bkgkey)
        for index, fitslice in enumerate(self.rod.slices):
             progress(index, fitslice)
        pd.close()

    def fit(self):
        if self.rod:
            self.rod.fit()
            self.plot()

    def fit_all(self):
        pd = QtGui.QProgressDialog('Performing fit...', 'Cancel', 0, len(self.rod.slices))
        pd.setWindowModality(QtCore.Qt.WindowModal)
        pd.show()
        def progress(i):
            pd.setValue(i)
            if pd.wasCanceled():
                raise KeyboardInterrupt
            QtGui.QApplication.processEvents()
            self.rod.fit(i)
        for index in range(len(self.rod.slices)):
             progress(index)
        pd.close()
        self.plot()


    def export(self, filename):
        self.rod.save_fit()
        self.rod.save_int()
        #extension = os.path.splitext(filename)[-1] 
        #if extension == '.txt':
        #    pass
        #else:
        #    self.error.showMessage("{0} is not a valid extension".format(extension))

    def tofile(self, filename):
        outdict = dict()
        outdict['filename'] = self.filename
        outdict['keys'] = list()
        outdict['values'] = list()
        outdict['variance'] = list()
        outdict['succes'] = list()
        outdict['loc'] = list()
        outdict['fitfunction'] = self.function_box.currentIndex()
        outdict['roi'] = self.intwidget.tolist()
        outdict['currentkey'] = list(self.rod.key)

        for key in self.roddict:
           outdict['keys'].append(key)
           result, succes, locs, variance = self.roddict[key].tolist()
           outdict['values'].append(result)
           outdict['variance'].append(variance)
           outdict['succes'].append(succes)
           outdict['loc'].append(locs)

        print outdict

        with open(filename, 'w') as fp:
            json.dump(outdict, fp)

    @classmethod
    def fromfile(cls, filename = None, parent = None):
        if filename == None:
            filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open Project', '.', '*.fit'))        
        try:
            with open(filename, 'r') as fp:
                dict = json.load(fp)
        except IOError as e:
            raise self.error.showMessage("unable to open '{0}' as project file (original error: {1!r})".format(filename, e))

        widget = cls(dict['filename'], parent = parent)
        for keyindex, key in enumerate(dict['keys']):
            widget.roddict[tuple(key)] = FitRod(dict['filename'], key[1], key[0])
            for sliceindex, fitslice in enumerate(widget.roddict[tuple(key)].slices):
                if dict['values'][keyindex][sliceindex] != None:
                    fitslice.result = numpy.array(dict['values'][keyindex][sliceindex])
                if dict['variance'][keyindex][sliceindex] != None:
                    fitslice.variance = numpy.array(dict['variance'][keyindex][sliceindex])
                fitslice.succes = dict['succes'][keyindex][sliceindex]
                fitslice.loc = dict['loc'][keyindex][sliceindex]
            
        if 'currentkey' in dict.keys():
            currentkey = dict['currentkey']
        else:
            currentkey = key
        widget.function_box.setCurrentIndex(dict['fitfunction'])
        widget.fitfunctionchanged(dict['fitfunction'])
        widget.resolution_axis.setCurrentIndex(currentkey[1])
        widget.resolution_line.setText(currentkey[0])
        widget.intwidget.values_from_list(dict['roi'])
        widget.set_res()

        return widget

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
        self.key = (str(resolution), axindex)


        if float(resolution) < ax.res:
            raise ValueError('interval {0} to low, minimum interval is {1}'.format(resolution, ax.res))

        mi, ma = ax.min, ax.max
        bins = numpy.linspace(mi, ma, numpy.ceil(1 / numpy.float(resolution) * (ma - mi)) + 1)

        for start, stop in zip(bins[:-1], bins[1:]):
            k = [slice(None) for i in self.axes]
            k[axindex] = slice(start, stop)
            coord = (start + stop) / 2
            self.slices.append(FitSlice(k, coord))

    def get_space(self, index = None):
        if index == None:
            return BINoculars.space.Space.fromfile(self.filename, self.slices[self.index].key).project(self.axis)
        else:
            return BINoculars.space.Space.fromfile(self.filename, self.slices[index].key).project(self.axis)

    def get_slice(self):
        return self.slices[self.index]

    def set_index(self,index):
        self.index = index

    def set_function(self, function):
        self.function = function

    def succesful(self):
        succesful = list()
        for fitslice in self.slices:
            if fitslice.succes:
                succesful.append(fitslice)
        return succesful

    def integrate(self, fitslice, space,  key, bkgkeys):        
        if fitslice.loc != None:
            intensity = interpolate(space[key]).flatten()
            bkg = numpy.hstack(space[bkgkey].get_masked().compressed() for bkgkey in bkgkeys)
            if numpy.alen(bkg) == 0 or numpy.alen(intensity) == 0:
                return numpy.nan
            else:
                return intensity.sum() - numpy.alen(intensity) / numpy.alen(bkg) * bkg.sum()

    def fit(self, index = None):
        if not index:
            fitslice = self.get_slice()
        else:
            fitslice = self.slices[index]

        space = BINoculars.space.Space.fromfile(self.filename, fitslice.key).project(self.axis)
        if not len(space.get_masked().compressed()) == 0:
            fit = self.function(space, loc = fitslice.loc)
            fitslice.fitdata = fit.fitdata
            fitslice.variance = fit.variance
            fitslice.result = fit.result
            xdata, ydata, cxdata, cydata = self.function._prepare(space)
            without_bkg = fit.result.copy()
            without_bkg[-3:] = 0
            nobkgdat = self.function.func(xdata, without_bkg)
            print without_bkg
            print nobkgdat.sum()
            fitslice.sum = nobkgdat.sum()
            print fitslice.sum
            fitslice.succes = True

    def tolist(self):
        results = list()
        succes = list()
        locs = list()
        variance = list()
        for fitslice in self.slices:
            if hasattr(fitslice, 'result'):
                results.append(list(fitslice.result))
                variance.append(list(fitslice.variance))
                succes.append(fitslice.succes)
            else:
                results.append(None)
                variance.append(None)
                succes.append(False)

        for fitslice in self.slices:
            locs.append(fitslice.loc)
        return results, succes, locs, variance

    def fit_loc(self, indices):
        deg = 4
        locdict = {}
        locx = list(fitslice.coord for fitslice in self.slices)
        for index in indices:
            x = list()
            y = list()
            w = list()
            for fitslice in self.succesful():
                x.append(fitslice.coord)
                y.append(fitslice.result[index])
                w.append(numpy.log(1 / fitslice.variance[index]))
            w = numpy.array(w)
            w[w == numpy.inf] = 0
            w = numpy.nan_to_num(w)
            w[w < 0] = 0
            if len(x) > 0:
                c = numpy.polynomial.polynomial.polyfit(x, y, deg, w = w)
                newy = numpy.polynomial.polynomial.polyval(locx, c)
                locdict[index] = newy
        if len(locdict) > 0:
            for i, fitslice in enumerate(self.slices):
                fitslice.loc = list(locdict[index][i] for index in indices)

    def save_fit(self):
        numpy.save('fit.npy', numpy.vstack([sl.coord, sl.sum] for sl in self.succesful()))

    def save_int(self):
        numpy.save('int.npy', numpy.vstack([sl.coord, sl.intensity] for sl in self.succesful()))
            
class FitSlice(object):
    def __init__(self, key, coord):
        self.key = key
        self.coord = coord
        self.loc = None
        self.succes = False

def short_filename(filename):
    return filename.split('/')[-1].split('.')[0]

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.resize(1000, 600)
    #main.loadproject('/home/willem/Documents/PhD/hc1151/analysis/highresrod/test4.fit')
    #main.plot_widget.set_res(axis = 'l')
    main.show()

    
    sys.exit(app.exec_())






