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

        self.config_widget = ConfigWidget()
        self.mask_widget = MaskWidget()

                  
        tab_widget.addTab(self.mask_widget, "Show images") 
        tab_widget.addTab(self.config_widget, "Set Configuration File") 

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
        
        

class ConfigWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ConfigWidget, self).__init__(parent)

        vbox = QtGui.QVBoxLayout()
        buttonhbox = QtGui.QHBoxLayout() 
        hbox = QtGui.QHBoxLayout() 
        left = QtGui.QVBoxLayout()
        middle = QtGui.QVBoxLayout()
        right = QtGui.QVBoxLayout()

        self.button_fromspec = QtGui.QPushButton('from specfile')
        self.button_fromspec.clicked.connect(self.fromspecfile)

        self.button_saveconfig = QtGui.QPushButton('save')
        self.button_saveconfig.clicked.connect(self.saveconfig)

        self.linelabel = QtGui.QLabel('scans to process')
        self.line = QtGui.QLineEdit()

        buttonhbox.addWidget(self.button_fromspec)
        buttonhbox.addWidget(self.button_saveconfig)
        buttonhbox.addWidget(self.linelabel)
        buttonhbox.addWidget(self.line)

        vbox.addLayout(buttonhbox)

        self.inputlabel = QtGui.QLabel(self)
        self.inputlabel.setText('input')
        self.inputtable = QtGui.QTableWidget()
        self.inputtable.dragEnabled()
        self.inputtable.setColumnCount(2)
        self.inputtable.setHorizontalHeaderLabels(['name','value'])
        self.inputtable.setMinimumSize(350, 0)

        self.dispatcherlabel = QtGui.QLabel(self)
        self.dispatcherlabel.setText('dispatcher')
        self.dispatchertable = QtGui.QTableWidget()
        self.dispatchertable.dragEnabled()
        self.dispatchertable.setColumnCount(2)
        self.dispatchertable.setHorizontalHeaderLabels(['name','value'])
        self.dispatchertable.setMinimumSize(350, 0)

        self.projectionlabel = QtGui.QLabel(self)
        self.projectionlabel.setText('projection')
        self.projectiontable = QtGui.QTableWidget()
        self.projectiontable.dragEnabled()
        self.projectiontable.setColumnCount(2)
        self.projectiontable.setHorizontalHeaderLabels(['name','value'])
        self.projectiontable.setMinimumSize(350, 0)

        left.addWidget(self.inputlabel)
        left.addWidget(self.inputtable)
        middle.addWidget(self.dispatcherlabel)
        middle.addWidget(self.dispatchertable)
        right.addWidget(self.projectionlabel)
        right.addWidget(self.projectiontable)

        hbox.addLayout(left)
        hbox.addLayout(middle)
        hbox.addLayout(right)

        vbox.addLayout(hbox)
        self.setLayout(vbox)


    def set_config_from_file(self, config):
        self.inputtable.setRowCount(len(config.input.keys()))
        for index in range(len(config.input.keys())):
            self.inputtable.setItem(index, 0, QtGui.QTableWidgetItem(str(config.input.keys()[index])))
            self.inputtable.setItem(index, 1, QtGui.QTableWidgetItem(str(config.input.values()[index])))
        self.dispatchertable.setRowCount(len(config.dispatcher.keys()))
        for index in range(len(config.dispatcher.keys())):
            self.dispatchertable.setItem(index, 0, QtGui.QTableWidgetItem(str(config.dispatcher.keys()[index])))
            self.dispatchertable.setItem(index, 1, QtGui.QTableWidgetItem(str(config.dispatcher.values()[index])))
        self.projectiontable.setRowCount(len(config.projection.keys()))
        for index in range(len(config.projection.keys())):
            self.projectiontable.setItem(index, 0, QtGui.QTableWidgetItem(str(config.projection.keys()[index])))
            self.projectiontable.setItem(index, 1, QtGui.QTableWidgetItem(str(config.projection.values()[index])))

    def fromspecfile(self):
        pass

    def saveconfig(self):
        pass


class MaskWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(MaskWidget, self).__init__(parent)

        self.spec_widget = QSpecFileWidget.QSpecFileWidget()
        self.browser_widget = StackBrowser.StackBrowser()

        self.button_show = QtGui.QPushButton('show')
        #self.button_show.clicked.connect(self.set_stack)

        self.button_refresh = QtGui.QPushButton('refresh')
        #self.button_refresh.clicked.connect(self.refresh_spec)

        hbox = QtGui.QHBoxLayout() 
        left = QtGui.QVBoxLayout()
        right = QtGui.QVBoxLayout()

        right.addWidget(self.browser_widget)

        top = QtGui.QHBoxLayout()
        top.addWidget(self.button_show)
        top.addWidget(self.button_refresh)
        left.addLayout(top)

        left.addWidget(self.spec_widget)

        hbox.addLayout(left)
        hbox.addLayout(right)

        self.setLayout(hbox)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())





