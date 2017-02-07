# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:00:12 2017

@author: jesse
"""

import numpy as np

class BinFile:
    def __init__(self, filename):
        self.totalChannels = 8
        self.filename = filename
        self.dataType = '<d'
        self.data = [0]
        
        #filename = '4031B_45deg_Day1_data.bin'
        #totalChannels = 8

    def setTotalChannels(self, n):
        self.totalChannels = n

    def setFilename(self, name):
        self.filename = name

    def setDataType(self, dt):
        self.dataType = dt

    def open_bin(self):
        
        self.data = np.fromfile(str(self.filename), dtype = np.dtype(self.dataType))
        #truncate the data file to handle bad input
        while(len(self.data) % self.totalChannels != 0):
            self.data = self.data[0:len(self.data)-1]
        self.data = self.data.reshape(len(self.data)/self.totalChannels, self.totalChannels)
        self.data = np.transpose(self.data)
        