import pyqtgraph as pg
from pyqtgraph import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from time import sleep
from sys import argv

pg.mkQApp()

A = 0.5
B = 1 - A
NFFT = 512
RADIO_CENTER_FREQ_MHZ = 160.45
RADIO_SAMPLE_RATE_MHZ = 0.768
FFT_FREQS = fftshift(fftfreq(NFFT,1/RADIO_SAMPLE_RATE_MHZ)+RADIO_CENTER_FREQ_MHZ)
FFT_FREQ_SPACING_MHZ = FFT_FREQS[int(NFFT/2)+1] - FFT_FREQS[int(NFFT/2)]

def freq_to_fft (fmhz):
    return int((fmhz - FFT_FREQS[0])/FFT_FREQ_SPACING_MHZ)

TAG_0_FREQ_MHZ = 160.12
TAG_SPACING_MHZ = 0.01

def freq_to_tag (fmhz):
    return int((fmhz - TAG_0_FREQ_MHZ)/TAG_SPACING_MHZ)

def tag_to_freq (tag):
    return TAG_0_FREQ_MHZ + tag * TAG_SPACING_MHZ

def tag_sum_range (tag):
    n1 = freq_to_fft(tag_to_freq (tag) - TAG_SPACING_MHZ/2)
    n2 = freq_to_fft(tag_to_freq (tag) + TAG_SPACING_MHZ/2)
    return (n1,n2,1.0/(n2-n1))

class DataSource (QtCore.QThread):

    onData = QtCore.pyqtSignal (tuple)

    def __init__ (self, data, frame_width=NFFT, sample_rate_mhz=0.768, center_freq_mhz=160.45):
        super().__init__()
        self.window = np.kaiser (frame_width,10)
        self.data = data
        self.frame_width = frame_width
        self.frame_rate = frame_width / sample_rate_mhz / 1000000
        self.freqs = fftshift(fftfreq(frame_width,1/sample_rate_mhz)+center_freq_mhz)
        self.bin = tag_sum_range(freq_to_tag(160.483))
        self.p1 = np.zeros(frame_width,np.float32)
        self.power = np.zeros(frame_width,np.float32)
        self.decimate = 10
        self.count = 0

    def run (self):
        nf = 0
        while nf < len(data):
            iq = data[nf:nf+self.frame_width] * self.window
            frm = fftshift(fft(iq))
            pwr = 10.0*np.log10((frm*np.conj(frm)).real)
            self.p1 = A*self.p1 + B*pwr
            self.power = A*self.power + B*self.p1
            self.count = self.count + 1
            if self.count > self.decimate:
                self.onData.emit ((self.freqs,self.power))
                chn = self.power[self.bin[0]:self.bin[1]]
                print(np.max(chn)-np.mean(chn))
                self.count = 0
            nf = nf + self.frame_width
            sleep (self.frame_rate)
            
class SpectrumLineGraph (pg.PlotWidget):

    def __init__ (self):
        super().__init__()
        self.pen = pg.mkPen (color=(0,255,0))
        self.dataLine = self.plot()
        self.viewport().setAttribute (Qt.WidgetAttribute.WA_AcceptTouchEvents,False)
        self.setDefaultPadding (0)
        self.setYRange (-10,-80) 
        self.showGrid (x=True,y=True,alpha=0.7)
        self.setMouseEnabled (x=False,y=False)
        self.hideButtons ()

    def redraw (self,data):
        self.setXRange (data[0][0],data[0][-1])
        self.dataLine.setData (data[0],data[1],pen=self.pen)

hideAxis = """
    QWidget {
        color : "black";
    }
"""
class WaterFallPlot (pg.PlotWidget):

    def __init__ (self,hight,width):
        super().__init__()
        #self.setStyleSheet (hideAxis)
        self.hight = hight
        self.width = width
        self.imageData = np.zeros((hight,width),np.float32)
        self.image = pg.ImageItem(axisOrder='row-major')
        self.addItem (self.image)
        self.image.setLevels((-110,-30))
        self.setMouseEnabled (x=False,y=False)
        self.setDefaultPadding (0)
        self.getAxis('bottom').setTextPen(QtGui.QPen())
        self.getAxis('left').setTextPen(QtGui.QPen())

    def add_data (self, line):
        self.imageData = np.roll(self.imageData,shift=-1,axis=0)
        self.imageData[0] = line
        self.image.setImage(self.imageData,autoLevel=False)

    def redraw (self, data):
        if len(data[1]) != self.width:
            self.width = len(data[1])
            self.imageData = np.zeros ((self.hight,self.width),np.float32)
        self.add_data (data[1])

class SpectrumDisplay (QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setLayout (QtWidgets.QVBoxLayout())
        layout = QtWidgets.QVBoxLayout()
        self.line_graph = SpectrumLineGraph ()
        self.waterfall = WaterFallPlot (500,NFFT)
        layout.addWidget (self.line_graph)
        layout.addWidget (self.waterfall)
        self.layout().addLayout (layout)

class MainWindow (QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.display = SpectrumDisplay()
        self.setCentralWidget (self.display)
        menuBar = QtWidgets.QMenuBar()
        file_menu = QtWidgets.QMenu ("&File",self)
        file_menu.addAction("Open")
        menuBar.addMenu (file_menu)
        self.setMenuBar (menuBar)

    def redraw (self,data):
        self.display.line_graph.redraw(data)
        self.display.waterfall.redraw(data)

if __name__ == '__main__':
    if argv[1]:
        data = np.fromfile(argv[1],dtype=np.complex64)
    else:
        data = np.fromfile ('res_to_mtn_ct_2.fc32',dtype=np.complex64)
    source = DataSource (data)
    display = MainWindow()
    source.onData.connect (display.redraw)
    display.show ()
    source.start()
    pg.exec ()
