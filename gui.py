import sys
import os
import glob
from PyQt4 import QtGui, QtCore, Qt
from PyMca import QSpecFileWidget, QDataSource, StackBrowser, StackSelector
import BINoculars.main, BINoculars.space, BINoculars.plot
import numpy
import json

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QTAgg
import matplotlib.figure, matplotlib.image



#RangeSlider is taken from https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
class RangeSlider(QtGui.QSlider):
    """ A slider for ranges.
    
        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.
        
        This class emits the same signals as the QSlider base class, with the 
        exception of valueChanged
    """
    def __init__(self, *args):
        super(RangeSlider, self).__init__(*args)
        
        self._low = self.minimum()
        self._high = self.maximum()
        
        self.pressed_control = QtGui.QStyle.SC_None
        self.hover_control = QtGui.QStyle.SC_None
        self.click_offset = 0
        
        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()
        
        
    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp

        painter = QtGui.QPainter(self)
        style = QtGui.QApplication.style() 
        
        for i, value in enumerate([self._low, self._high]):
            opt = QtGui.QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QtGui.QStyle.SC_SliderHandle#QtGui.QStyle.SC_SliderGroove | QtGui.QStyle.SC_SliderHandle
            else:
                opt.subControls = QtGui.QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QtGui.QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QtGui.QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value                                  
            style.drawComplexControl(QtGui.QStyle.CC_Slider, opt, painter, self)
            
        
    def mousePressEvent(self, event):
        event.accept()
        
        style = QtGui.QApplication.style()
        button = event.button()
        
        # In a normal slider control, when the user clicks on a point in the 
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts
                
        if button:
            opt = QtGui.QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1
            
            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value                
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit
                    
                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break

            if self.active_slider < 0:
                self.pressed_control = QtGui.QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()


    def mouseReleaseEvent(self, event):
        self.emit(QtCore.SIGNAL('sliderReleased()'))

                                
    def mouseMoveEvent(self, event):
        if self.pressed_control != QtGui.QStyle.SC_SliderHandle:
            event.ignore()
            return
        
        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)
        
        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff            
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos

        self.update()

        self.emit(QtCore.SIGNAL('sliderMoved(int)'), new_pos)
            
    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()
           
           
    def __pixelPosToRangeValue(self, pos):
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QtGui.QApplication.style()
        
        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)
        
        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1
            
        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos-slider_min, slider_max-slider_min,
                                             opt.upsideDown)



class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        newproject = QtGui.QAction("New project", self)  
        newproject.triggered.connect(self.newproject)

        loadproject = QtGui.QAction("Open project", self)  
        loadproject.triggered.connect(self.loadproject)

        saveproject = QtGui.QAction("Save project", self)  
        saveproject.triggered.connect(self.saveproject)

        addspace = QtGui.QAction("Import space", self)  
        addspace.triggered.connect(self.add_to_project)

        menu_bar = QtGui.QMenuBar() 
        file = menu_bar.addMenu("&File") 
        file.addAction(newproject) 
        file.addAction(loadproject) 
        file.addAction(saveproject)
        file.addAction(addspace) 

        self.tab_widget = QtGui.QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.tab_widget.removeTab)

        self.statusbar	= QtGui.QStatusBar()

        self.setCentralWidget(self.tab_widget)
        self.setMenuBar(menu_bar)
        self.setStatusBar(self.statusbar)

    def newproject(self):
        self.tab_widget.addTab(ProjectWidget([], parent=self), 'New Project')
            
    def loadproject(self, filename = None):
        if not filename:
            dialog = QtGui.QFileDialog(self, "Load project");
            dialog.setFilter('BINoculars project file (*.proj)');
            dialog.setFileMode(QtGui.QFileDialog.ExistingFiles);
            dialog.setAcceptMode(QtGui.QFileDialog.AcceptOpen);
            if not dialog.exec_():
                return
            fname = dialog.selectedFiles()
            if not fname:
                return
            for name in fname:
                try:
                    widget = ProjectWidget.fromfile(str(name), parent = self)
                    self.tab_widget.addTab(widget, short_filename(str(name)))
                except Exception as e:
                    QtGui.QMessageBox.critical(self, 'Load project', 'Unable to load project from {}: {}'.format(fname, e))
        else:
            widget = ProjectWidget.fromfile(filename, parent = self)
            self.tab_widget.addTab(widget, short_filename(filename))

    def saveproject(self):
        widget = self.tab_widget.currentWidget()
        dialog = QtGui.QFileDialog(self, "Save project");
        dialog.setFilter('BINoculars project file (*.proj)');
        dialog.setDefaultSuffix('proj');
        dialog.setFileMode(QtGui.QFileDialog.AnyFile);
        dialog.setAcceptMode(QtGui.QFileDialog.AcceptSave);
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()[0]
        if not fname:
            return
        try:
            index = self.tab_widget.currentIndex()
            self.tab_widget.setTabText(index, short_filename(fname))
            widget.tofile(fname)
        except Exception as e:
            QtGui.QMessageBox.critical(self, 'Save project', 'Unable to save project to {}: {}'.format(fname, e))

    def add_to_project(self):
        dialog = QtGui.QFileDialog(self, "Import spaces");
        dialog.setFilter('BINoculars space file (*.hdf5)');
        dialog.setFileMode(QtGui.QFileDialog.ExistingFiles);
        dialog.setAcceptMode(QtGui.QFileDialog.AcceptOpen);
        if not dialog.exec_():
            return
        fname = dialog.selectedFiles()
        if not fname:
            return
        for name in fname:
            try:
                widget = self.tab_widget.currentWidget()
                widget.addspace(str(name))
            except Exception as e:
                QtGui.QMessageBox.critical(self, 'Import spaces', 'Unable to import space {}: {}'.format(fname, e))


class HiddenToolbar(NavigationToolbar2QTAgg):
    def __init__(self, show_coords, canvas):
        NavigationToolbar2QTAgg.__init__(self, canvas, None)
        self.show_coords = show_coords

    def mouse_move(self, event):
        self.show_coords(event)


class ProjectWidget(QtGui.QWidget):
    def __init__(self, filelist, key = None, projection = None, parent = None):
        super(ProjectWidget, self).__init__(parent)
        self.parent = parent

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = HiddenToolbar(self.show_coords, self.canvas)

        self.log = QtGui.QCheckBox('log', self)
        self.log.setChecked(True)
        QtCore.QObject.connect(self.log, QtCore.SIGNAL("stateChanged(int)"), self.plot)

        self.samerange = QtGui.QCheckBox('same intensity range', self)
        self.samerange.setChecked(False)
        QtCore.QObject.connect(self.samerange, QtCore.SIGNAL("stateChanged(int)"), self.update_colorbar)


        self.datarange = RangeSlider(Qt.Qt.Horizontal)
        self.datarange.setMinimum(0)
        self.datarange.setMaximum(250)
        self.datarange.setLow(0)
        self.datarange.setHigh(self.datarange.maximum())
        self.datarange.setTickPosition(QtGui.QSlider.TicksBelow)
        QtCore.QObject.connect(self.datarange, QtCore.SIGNAL('sliderMoved(int)'), self.update_colorbar)

        self.table = TableWidget(filelist)
        QtCore.QObject.connect(self.table, QtCore.SIGNAL('selectionError'), self.selectionerror)

        self.key = key
        self.projection = projection

        hbox = QtGui.QHBoxLayout() 
        left = QtGui.QVBoxLayout()
        right = QtGui.QVBoxLayout()

        self.button_save = QtGui.QPushButton('save image')
        self.button_save.clicked.connect(self.save)

        self.limitwidget = LimitWidget(self.table.plotaxes)
        QtCore.QObject.connect(self.limitwidget, QtCore.SIGNAL("keydict"), self.update_key)
        QtCore.QObject.connect(self.limitwidget, QtCore.SIGNAL("rangechange"), self.update_figure_range)
        QtCore.QObject.connect(self.table, QtCore.SIGNAL('plotaxesChanged'), self.plotaxes_changed)
        
        left.addWidget(self.button_save)

        radiobox =  QtGui.QHBoxLayout() 
        self.group = QtGui.QButtonGroup(self)
        for label in ['stack', 'grid']:
            rb = QtGui.QRadioButton(label, self)
            self.group.addButton(rb)
            radiobox.addWidget(rb)

        datarangebox = QtGui.QHBoxLayout() 
        datarangebox.addWidget(self.log)
        datarangebox.addWidget(self.samerange)

        left.addLayout(radiobox)
        left.addLayout(datarangebox)
        left.addWidget(self.datarange)

        left.addWidget(self.table)

        left.addWidget(self.limitwidget)
        right.addWidget(self.canvas)

        hbox.addLayout(left)
        hbox.addLayout(right)

        self.setLayout(hbox)
        self.table.select()

        
    def show_coords(self, event):
        plot = event.inaxes
        if hasattr(plot, 'space'):
            if plot.space.dimension == 2:
                labels = numpy.array([plot.get_xlabel(), plot.get_ylabel()])
                order = [plot.space.axes.index(label) for label in labels]
                labels = labels[order]                
                coords = numpy.array([event.xdata, event.ydata])[order] 
                rounded_coords = [ax[ax.get_index(coord)] for ax, coord in zip(plot.space.axes, coords)]
                intensity = '{:.2e}'.format(plot.space.get_value(list(coords)))
                self.parent.statusbar.showMessage('{0} = {1}, {2} = {3}, Intensity = {4}'.format(labels[0], rounded_coords[0] ,labels[1], rounded_coords[1], intensity))
            if plot.space.dimension == 1:
                xaxis = plot.space.axes[plot.space.axes.index(plot.get_xlabel())]
                if event.xdata in xaxis:
                     xcoord =  xaxis[xaxis.get_index(event.xdata)]
                     intensity = '{:.2e}'.format(event.ydata)
                     self.parent.statusbar.showMessage('{0} = {1}, Intensity = {2}'.format(xaxis.label, xcoord, intensity))

        

    def selectionerror(self, message):
        self.limitwidget.setDisabled(True)
        self.errormessage(message)

    def plotaxes_changed(self, plotaxes):
        self.limitwidget.setEnabled(True)
        self.limitwidget.axes_update(plotaxes)

    def update_key(self, input):
        self.key = input['key']
        self.projection = input['project']

        if len(self.limitwidget.sliders) - len(self.projection) == 1:
            self.datarange.setDisabled(True)
            self.samerange.setDisabled(True)
        elif len(self.limitwidget.sliders) - len(self.projection) == 2:
            self.datarange.setEnabled(True)
            self.samerange.setEnabled(True)
        self.plot()

    @staticmethod
    def restricted_key(key, axes):
        return tuple(ax.restrict(s) for s, ax in zip(key, axes))

    def get_norm(self, mi, ma):
        log = self.log.checkState()

        rangemin = self.datarange.low() * 1.0 / self.datarange.maximum()
        rangemax = self.datarange.high() * 1.0 / self.datarange.maximum()

        if log:
            power = 3
            vmin = mi + (ma - mi) * rangemin ** power
            vmax = mi + (ma - mi) * rangemax ** power
        else:
            vmin = mi + (ma - mi) * rangemin
            vmax = mi + (ma - mi) * rangemax

        if log:
            return matplotlib.colors.LogNorm(vmin, vmax)
        else:
            return matplotlib.colors.Normalize(vmin, vmax)

    def get_normlist(self):
        log = self.log.checkState()
        same = self.samerange.checkState()

        if same:
            return [self.get_norm(min(self.datamin), max(self.datamax))] * len(self.datamin)
        else:
            norm = []
            for i in range(len(self.datamin)):
                norm.append(self.get_norm(self.datamin[i], self.datamax[i]))
            return norm

    def plot(self):
        self.figure.clear()
        self.parent.statusbar.clearMessage()

        self.figure_images = []
        log = self.log.checkState()

        plotcount = len(self.table.selection)
        plotcolumns = int(numpy.ceil(numpy.sqrt(plotcount)))
        plotrows = int(numpy.ceil(float(plotcount) / plotcolumns))
        plotoption = None
        if self.group.checkedButton():
            plotoption = self.group.checkedButton().text()
        
        spaces = []
        for i, filename in enumerate(self.table.selection):
            axes = BINoculars.space.Axes.fromfile(filename)
            space = BINoculars.space.Space.fromfile(filename, key = self.restricted_key(self.key, axes))
            projection = [ax for ax in self.projection if ax in space.axes]
            if projection:
                space = space.project(*projection)
            if len(space.axes) > 2 or len(space.axes) == 0:
                self.errormessage('choose suitable number of projections, plotting only in 1D and 2D')
            spaces.append(space)

        self.datamin = []
        self.datamax = []
        for space in spaces:
            data = space.get_masked().compressed()
            if log:
                data = data[data > 0]
            self.datamin.append(data.min())
            self.datamax.append(data.max())
        norm = self.get_normlist()

        for i,space in enumerate(spaces):
            if plotcount > 1:
                if space.dimension == 1 and (plotoption == 'stack' or plotoption == None):
                    self.ax = self.figure.add_subplot(111)
                if space.dimension == 2 and plotoption != 'grid':
                    sys.stderr.write('warning: stack display not supported for multi-file-plotting, falling back to grid\n')
                    plotoption = 'grid'
                elif space.dimension > 3:
                    sys.stderr.write('error: cannot display 4 or higher dimensional data, use --project or --slice to decrease dimensionality\n')
                    sys.exit(1)
            else:
                 self.ax = self.figure.add_subplot(111)
            basename = os.path.splitext(os.path.basename(filename))[0]

            if plotoption == 'grid':
                self.ax = self.figure.add_subplot(plotrows, plotcolumns, i+1)
            self.ax.space = space
            im = BINoculars.plot.plot(space,self.figure, self.ax, log = log,label = basename, norm = norm[i])
            self.figure_images.append(im)

        #if plotcount > 1 and plotoption == 'stack':
        #    self.figure.legend()

        self.canvas.draw()

    def errormessage(self, message):
        self.figure.clear()
        self.canvas.draw()
        self.parent.statusbar.showMessage(message)

    def update_figure_range(self, key):
        for ax in self.figure.axes:
            xlabel, ylabel = ax.get_xlabel(), ax.get_ylabel()
            if xlabel in self.table.plotaxes:
                xindex = self.table.plotaxes.index(xlabel)
                ax.set_xlim(key[xindex][0], key[xindex][1])
            if ylabel in self.table.plotaxes:
                yindex = self.table.plotaxes.index(ylabel)
                ax.set_ylim(key[yindex][0], key[yindex][1])
        self.canvas.draw()

    def update_colorbar(self,value):
        normlist = self.get_normlist()
        for im,norm in zip(self.figure_images, normlist):
            im.set_norm(norm)
        self.canvas.draw()

    @staticmethod
    def key_to_str(key):
        return list([s.start, s.stop] for s in key)

    @staticmethod
    def str_to_key(s):
        return tuple(slice(float(key[0]), float(key[1])) for key in s)

    def tofile(self, filename = None):
        dict = {}
        dict['filelist'] = self.table.filelist
        dict['key'] = self.key_to_str(self.key)
        dict['projection'] = self.projection

        if filename == None:
            filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Save Project', '.'))

        with open(filename, 'w') as fp:
            json.dump(dict, fp)

    @classmethod
    def fromfile(cls, filename = None, parent = None):
        if filename == None:
            filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open Project', '.', '*.proj'))        
        try:
            with open(filename, 'r') as fp:
                dict = json.load(fp)
        except IOError as e:
            raise self.error.showMessage("unable to open '{0}' as project file (original error: {1!r})".format(filename, e))

        widget = cls(dict['filelist'], cls.str_to_key(dict['key']), dict['projection'], parent = parent)

        return widget
    
    def addspace(self,filename = None):
        if filename == None:
            filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open Project', '.', '*.hdf5'))
        self.table.addspace(filename)

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

def short_filename(filename):
    return filename.split('/')[-1].split('.')[0]

class TableWidget(QtGui.QWidget):
    def __init__(self, filelist = [],parent=None):
        super(TableWidget, self).__init__(parent)

        hbox = QtGui.QHBoxLayout()
        self.plotaxes = []

        self.table = QtGui.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['filename','labels', 'remove'])
        
        for index, width in enumerate([150,50,70]):
            self.table.setColumnWidth(index, width)

        for filename in filelist:
            self.addspace(filename, True)

        hbox.addWidget(self.table)
        self.setLayout(hbox)

    def addspace(self, filename, add = False):
        def remove_callback(filename):
            return lambda: self.remove(filename)

        index = self.table.rowCount()
        self.table.insertRow(index)

        axes = BINoculars.space.Axes.fromfile(filename) 

        checkboxwidget = QtGui.QCheckBox(short_filename(filename))
        checkboxwidget.setChecked(add or index == 0)
        checkboxwidget.filename = filename
        checkboxwidget.clicked.connect(self.select)
        self.table.setCellWidget(index,0, checkboxwidget)

        item = QtGui.QTableWidgetItem(','.join(list(ax.label.lower() for ax in axes)))
        self.table.setItem(index, 1, item)

        buttonwidget = QtGui.QPushButton('remove')
        buttonwidget.clicked.connect(remove_callback(filename))
        self.table.setCellWidget(index,2, buttonwidget)
        self.select()

    def remove(self, filename):
        table_filenames = list(self.table.cellWidget(index, 0).filename for index in range(self.table.rowCount()))
        for index, label in enumerate(table_filenames):
            if filename == label:
                self.table.removeRow(index)
        self.select()
        print 'removed: {0}'.format(filename)

    def select(self):
        self.selection = []
        for index in range(self.table.rowCount()):
            checkbox = self.table.cellWidget(index, 0)
            if checkbox.checkState():
                self.selection.append(checkbox.filename)

        if len(self.selection) > 0:
            axeslist = list(BINoculars.space.Axes.fromfile(filename) for filename in self.selection)
            first = axeslist[0]
            if all(set(ax.label for ax in first) == set(ax.label for ax in axes) for axes in axeslist):
                self.plotaxes = BINoculars.space.Axes(tuple(BINoculars.space.union_unequal_axes(ax[i] for ax in axeslist) for i in range(first.dimension)))
                self.emit(QtCore.SIGNAL('plotaxesChanged'), self.plotaxes)
            else:
                self.selection = []
                self.emit(QtCore.SIGNAL('selectionError'), 'labels of selected spaces not matching')
        else:
            self.emit(QtCore.SIGNAL('selectionError'), 'no spaces selected')

            
    @property
    def filelist(self):
        return list(self.table.cellWidget(index, 0).filename for index in range(self.table.rowCount()))


class LimitWidget(QtGui.QWidget):
    def __init__(self, axes, parent=None):
        super(LimitWidget, self).__init__(parent)

        self.initUI(axes)

    def initUI(self, axes):
        self.axes = axes

        self.sliders = list()
        self.qlabels = list()
        self.leftindicator = list()
        self.rightindicator = list()

        labels = list(ax.label for ax in axes)

        vbox = QtGui.QVBoxLayout()
        hbox = QtGui.QHBoxLayout()

        self.projectionlabel = QtGui.QLabel(self)
        self.projectionlabel.setText('projection along axis')
        
        vbox.addWidget(self.projectionlabel)

        self.checkbox = list()
        self.state = list()

        for label in labels:
            self.checkbox.append(QtGui.QCheckBox(label, self))
        for box in self.checkbox:
            self.state.append(box.checkState())
            hbox.addWidget(box)
            box.stateChanged.connect(self.update_checkbox)
        
        self.state = numpy.array(self.state, dtype = numpy.bool)
        self.init_checkbox()

        vbox.addLayout(hbox)
        

        for label in labels:
            self.qlabels.append(QtGui.QLabel(self))
            self.leftindicator.append(QtGui.QLineEdit(self))
            self.rightindicator.append(QtGui.QLineEdit(self))             
            self.sliders.append(RangeSlider(Qt.Qt.Horizontal))

        for index, label in enumerate(labels):
            box = QtGui.QHBoxLayout()
            box.addWidget(self.qlabels[index])
            box.addWidget(self.leftindicator[index])
            box.addWidget(self.sliders[index])
            box.addWidget(self.rightindicator[index])
            vbox.addLayout(box)

        for left in self.leftindicator:
            left.setMaximumWidth(50)
        for right in self.rightindicator:
            right.setMaximumWidth(50)

        for index, label in enumerate(labels):
            self.qlabels[index].setText(label)

        for index, ax in enumerate(axes):
            self.sliders[index].setMinimum(0)
            self.sliders[index].setMaximum(len(ax) - 1)
            self.sliders[index].setLow(0)
            self.sliders[index].setHigh(len(ax) - 1)
            self.sliders[index].setTickPosition(QtGui.QSlider.TicksBelow)

        self.update_lines()

        for slider in self.sliders:
            QtCore.QObject.connect(slider, QtCore.SIGNAL('sliderMoved(int)'), self.update_lines)
        for slider in self.sliders:
            QtCore.QObject.connect(slider, QtCore.SIGNAL('sliderReleased()'), self.send_signal)

        for line in self.leftindicator:
            line.editingFinished.connect(self.update_sliders_left)
        for line in self.rightindicator:
            line.editingFinished.connect(self.update_sliders_right)

        if self.layout() == None:
            self.setLayout(vbox)

    def update_lines(self, value = 0 ):
        for index, slider in enumerate(self.sliders):
            self.leftindicator[index].setText(str(self.axes[index][slider.low()]))
            self.rightindicator[index].setText(str(self.axes[index][slider.high()]))
        key = list((float(str(left.text())), float(str(right.text()))) for left, right in zip(self.leftindicator, self.rightindicator))
        self.emit(QtCore.SIGNAL('rangechange'), key)

    def send_signal(self):
        signal = {}
        key = ((float(str(left.text())), float(str(right.text()))) for left, right in zip(self.leftindicator, self.rightindicator))
        key = [left if left == right else slice(left, right, None) for left, right in key]
        project = []
        for ax, state in zip(self.axes, self.state):
            if state:
                project.append(ax.label)
        signal['project'] = project
        signal['key'] = key
        self.emit(QtCore.SIGNAL('keydict'), signal)
            
 
    def update_sliders_left(self):
        for ax, left, right , slider in zip(self.axes, self.leftindicator, self.rightindicator, self.sliders):
            try:
                leftvalue = ax.get_index(float(str(left.text())))
                rightvalue = ax.get_index(float(str(right.text())))
                if leftvalue >= slider.minimum() and leftvalue < rightvalue:
                    slider.setLow(leftvalue)
                else:
                    slider.setLow(rightvalue - 1)
            except ValueError:
                slider.setLow(0)
            left.setText(str(ax[slider.low()]))

    def update_sliders_right(self):
        for ax, left, right , slider in zip(self.axes, self.leftindicator, self.rightindicator, self.sliders):
            leftvalue = ax.get_index(float(str(left.text())))
            try:
                rightvalue = ax.get_index(float(str(right.text())))
                if rightvalue <= slider.maximum() and rightvalue > leftvalue:
                    slider.setHigh(rightvalue)
                else:
                    slider.setHigh(leftvalue + 1)
            except ValueError:
                slider.setHigh(len(ax) - 1)
            right.setText(str(ax[slider.high()]))

    def update_checkbox(self):
        self.state = list()
        for box in self.checkbox:
            self.state.append(box.checkState())
        self.send_signal()

    def init_checkbox(self):
        while numpy.alen(self.state) - self.state.sum() > 2:
             index = numpy.where(self.state == False)[-1]
             self.state[-1] = True     
        for box, state in zip(self.checkbox,self.state):
            box.setChecked(state)

    def axes_update(self, axes):
        if not set(ax.label for ax in self.axes) == set(ax.label for ax in axes):
            QtGui.QWidget().setLayout(self.layout())
            self.initUI(axes)
            self.send_signal()
        else:
            low = tuple(self.axes[index][slider.low()] for index, slider in enumerate(self.sliders))
            high = tuple(self.axes[index][slider.high()] for index, slider in enumerate(self.sliders))

            for index, ax in enumerate(axes):
                self.sliders[index].setMinimum(0)
                self.sliders[index].setMaximum(len(ax) - 1)

            self.axes = axes

            for index, slider in enumerate(self.sliders):
                self.leftindicator[index].setText(str(low[index]))
                self.rightindicator[index].setText(str(high[index]))

            self.update_sliders_left()
            self.update_sliders_right()

            self.send_signal()
            
    
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.resize(1000, 600)
    main.newproject()
    main.show()

    sys.exit(app.exec_())






