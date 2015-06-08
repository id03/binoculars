"""
BINoculars gui for data processing 
Created on 2015-06-04
author: Remy Nencib (remy.nencib@esrf.r)
"""

import sys
import os
import glob
from PyQt4 import QtGui, QtCore, Qt
import sys, os
import itertools
import inspect
import glob
import BINoculars.util, BINoculars.main
import time


class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.initUI()
        self.tab_widget = QtGui.QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        close = self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)

    def close_tab(self, tab):
        self.tab_widget.removeTab(tab)

    def initUI(self):
        openFile = QtGui.QAction('Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.ShowFile)

        saveFile = QtGui.QAction('Save', self)
        saveFile.setShortcut('Ctrl+S')
        saveFile.setStatusTip('Save File')
        saveFile.triggered.connect(self.Save)

        Create = QtGui.QAction('Create', self)
        Create.setStatusTip('Create Configfile')
        Create.triggered.connect(self.New_Config)
         
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(saveFile)
        fileMenu = menubar.addMenu('&New Configfile')
        fileMenu.addAction(Create)
        fileMenu = menubar.addMenu('&HELP')

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background,QtCore.Qt.gray)
        self.setPalette(palette)
        self.setGeometry(250, 100,500,500)
        self.setWindowTitle('Binoculars processgui')
        self.setWindowIcon(QtGui.QIcon('binoculars.png'))
        self.show()

    def ShowFile(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '')
        for F in filename.split('/') :
            NameFile = []
            NameFile.append(F)
        NameFile.reverse()
        self.tab_widget.addTab(Conf_Tab(self),NameFile[0])
        widget = self.tab_widget.currentWidget()
        widget.read_data(filename)


    def Save(self):
        filename = QtGui.QFileDialog().getSaveFileName(self, 'Save', '', '*.txt')
        widget = self.tab_widget.currentWidget() 
        widget.save(filename) 
        
    def New_Config(self):
        self.tab_widget.addTab(Conf_Tab(self),'New configfile')

#----------------------------------------------------------------------------------------------------
#-----------------------------------------CREATE TABLE-----------------------------------------------
class Table(QtGui.QWidget):
    def __init__(self, parent = None):
        super(Table, self).__init__()
        
        # create a QTableWidget
        self.table = QtGui.QTableWidget(1, 3, self)
        self.table.setHorizontalHeaderLabels(['Parameter', 'Value','Comment'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        
        #create combobox
        self.combobox = QtGui.QComboBox()
        #add items
        cell = QtGui.QTableWidgetItem(QtCore.QString("type"))
        cell2 = QtGui.QTableWidgetItem(QtCore.QString(""))
        self.table.setItem(0, 0, cell)
        self.table.setCellWidget(0, 1, self.combobox)
        self.table.setItem(0, 2,cell2)

        self.btn_add_row = QtGui.QPushButton('+', self)
        self.connect(self.btn_add_row, QtCore.SIGNAL('clicked()'), self.add_row)
        self.buttonRemove = QtGui.QPushButton('-',self)
        self.connect(self.buttonRemove, QtCore.SIGNAL("clicked()"), self.remove) 

        layout = QtGui.QGridLayout()
        layout.addWidget(self.table,0,0,3,10)
        layout.addWidget(self.btn_add_row,0,11)
        layout.addWidget(self.buttonRemove,1,11)
        self.setLayout(layout)

    def add_row(self):
        self.table.insertRow(self.table.rowCount())
    
    def remove(self):
        self.table.removeRow(self.table.currentRow()) 

    def get_keys(self):
        return list(self.table.item(index,0).text() for index in range(self.table.rowCount())) 


    def getParam(self):
        for index in range(self.table.rowCount()):
            key = str(self.table.item(index,0).text()) 
            comment = str(self.table.item(index, 2).text())
            if self.table.item(index,1):
                value = str(self.table.item(index, 1).text())
            else:
                value = str(self.table.cellWidget(index, 1).currentText())
            if self.table.item == None:
                value = str(self.table.item(index,1).text(""))
            yield key, value, comment
        
    def addData(self, data):
        for item in data:
            if item[0] == 'type':
                box = self.table.cellWidget(0,1)
                box.setCurrentIndex(box.findText(item[1], QtCore.Qt.MatchFixedString))
                newitem = QtGui.QTableWidgetItem(item[2])
                self.table.setItem(0, 2, newitem)
            else: 
                self.add_row()
                row = self.table.rowCount()
                for col in range(self.table.columnCount()):
                    newitem = QtGui.QTableWidgetItem(item[col])
                    self.table.setItem(row -1, col, newitem)

    def addDataConf(self, items):
        keys = self.get_keys()
        newconfigs = list([item[0], '', item[1]] for item in items if item[0] not in keys)
        self.addData(newconfigs)
                
    def add_to_combo(self, items):
        self.combobox.clear()
        self.combobox.addItems(items)
    

#----------------------------------------------------------------------------------------------------
#-----------------------------------------CREATE CONFIG----------------------------------------------
class Conf_Tab(QtGui.QWidget):
    def __init__(self, parent = None):

        super(Conf_Tab,self).__init__()
        self.Dis = Table()
        self.Inp = Table()
        self.Pro = Table()

        label1 = QtGui.QLabel('<strong>Dispatcher</strong>')
        label2 = QtGui.QLabel('<strong>Input</strong>')
        label3 = QtGui.QLabel('<strong>Projection<strong>')

        self.select = QtGui.QComboBox()
        backends = list(backend.lower() for backend in BINoculars.util.get_backends())
        self.select.addItems(QtCore.QStringList(backends))
        self.start = QtGui.QPushButton('run')
        self.connect(self.start, QtCore.SIGNAL("clicked()"), self.run)
        self.scan = QtGui.QLineEdit()
        self.start.setStyleSheet("background-color: darkred")

        Layout = QtGui.QGridLayout()
        Layout.addWidget(self.select,0,1)
        Layout.addWidget(label1,1,1)
        Layout.addWidget(self.Dis,2,1)
        Layout.addWidget(label2,3,1)
        Layout.addWidget(self.Inp,4,1)
        Layout.addWidget(label3,5,1)
        Layout.addWidget(self.Pro,6,1) 
        Layout.addWidget(self.start,7,0)
        Layout.addWidget(self.scan,7,1)
        self.setLayout(Layout)
 
        self.Dis.add_to_combo(QtCore.QStringList(BINoculars.util.get_dispatchers()))
        self.select.activated['QString'].connect(self.DataCombo)
        self.Inp.combobox.activated['QString'].connect(self.DataTableInp)
        self.Pro.combobox.activated['QString'].connect(self.DataTableInpPro)
        self.Dis.combobox.activated['QString'].connect(self.DataTableInpDis)
        

    def DataCombo(self,text):
        self.Inp.add_to_combo(QtCore.QStringList(BINoculars.util.get_inputs(str(text))))
        self.Pro.add_to_combo(QtCore.QStringList(BINoculars.util.get_projections(str(text))))

    def DataTableInp (self,text):
        backend = str(self.select.currentText())
        inp = BINoculars.util.get_input_configkeys(backend, str(self.Inp.combobox.currentText()))
        self.Inp.addDataConf(inp)

    def DataTableInpPro (self,text):
        backend = str(self.select.currentText())
        proj = BINoculars.util.get_projection_configkeys(backend, str(self.Pro.combobox.currentText()))
        self.Pro.addDataConf(proj)

    def DataTableInpDis (self,text):
        backend = str(self.select.currentText())
        disp = BINoculars.util.get_dispatcher_configkeys(str(self.Dis.combobox.currentText()))
        self.Dis.addDataConf(disp)

 
    def save(self, filename): 
        with open(filename, 'w') as fp:
            fp.write('[dispatcher]\n')
            for key, value, comment in self.Dis.getParam():# cycles over the iterator object
                fp.write('{0} = {1} #{2}\n'.format(key, value, comment))
            fp.write('[input]\n')
            for key, value, comment in self.Inp.getParam():
                if key == 'type':
                    value = '{0}:{1}'.format(self.select.currentText(),value)
                fp.write('{0} = {1} #{2}\n'.format(key, value, comment))
            fp.write('[projection]\n')
            for key, value, comment in self.Pro.getParam():
                if key == 'type':
                    value = '{0}:{1}'.format(self.select.currentText(),value)
                fp.write('{0} = {1} #{2}\n'.format(key, value, comment))

    def get_configobj(self):

        inDis = dict((key, value) for key, value, comment in self.Dis.getParam())
        inInp = {}
        inPro = {}

        for key, value, comment in self.Inp.getParam():
            if key == 'type':
                value = '{0}:{1}'.format(str(self.select.currentText()).strip(),value)
            inInp[key] = value   

        for key, value, comment in self.Pro.getParam():
            if key == 'type':
                value = '{0}:{1}'.format(str(self.select.currentText()).strip(),value)
            inPro[key] = value

        cfg = BINoculars.util.ConfigFile('processgui {0}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
        setattr(cfg, 'input', inInp)
        setattr(cfg, 'dispatcher', inDis)
        setattr(cfg, 'projection', inPro)

        print inInp

        return cfg


    def read_data(self,filename):
        with open(filename, 'r') as inf:
            lines = inf.readlines()
 
        data = {'dispatcher': [], 'input': [], 'projection': []}
        for line in lines:
            line = line.strip('\n')
            if '[dispatcher]' in line:
                key = 'dispatcher'
            elif '[input]' in line:
                key = 'input'
            elif '[projection]' in line: 
                key = 'projection'
            else:
                if '#' in line:
                    index = line.index('#')
                    caput = line[:index]
                    cauda = line[index:]
                else:
                    caput = line
                    cauda = ''
                if '=' in caput:
                    name, value = caput.split('=')
                    if name.strip(' ') == 'type' and ':' in value:
                        backend, value = value.strip(' ').split(':')
                    data[key].append([name.strip(' '), value.strip(' '), cauda.strip(' ')])

        self.select.setCurrentIndex(self.select.findText(backend, QtCore.Qt.MatchFixedString))
        self.DataCombo(backend)

        for key in data:
            if key == 'dispatcher':
                self.Dis.addData(data[key])
            elif key == 'input':
                self.Inp.addData(data[key])
            elif key == 'projection':
                self.Pro.addData(data[key])
                
    def run(self):
        command = [str(self.scan.text())]
        cfg = self.get_configobj()

        print 'Command: {0}'.format(command)
        print cfg

        BINoculars.main.Main.from_object(cfg, command)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())





