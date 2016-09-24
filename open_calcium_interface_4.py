# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:53:42 2016

@author: Jesse
"""
import numpy as np
import matplotlib.pyplot as plt
import tkFileDialog
import csv
import Tkinter as tk
import scipy.stats as stats
#from scipy import signal
from scipy.signal import butter, lfilter#, freqz
import copy
from pylab import *
import collections

#plt.rcParams["font.family"] = "helvetica"

class Cell():
    def __init__(self, name, intensity_data):
        self.name = name
        self.responsive = False
        
        self.intensity = intensity_data
        
        self.flips = dict()
        self.flops = dict()
        self.gray = dict()
        
        self.blocks = dict()
        
        
class FileHandler():
    def __init__(self):
        self.strobe_up = list()
        self.strobe_down = list()
        self.onset_up = list()
        self.onset_down = list()
        
        self.stim_up = list()
        self.stim_down = list()
        
        self.Bframe_up = list()
        self.Bframe_down = list()
        
    
    #Opens event channel Data from csv
    def open_event_csv(self):
        files = tkFileDialog.askopenfilenames()
        if files:
            files = list(files)
            with open(files[0], 'rU') as csvfile:
                reader = csv.reader(csvfile, delimiter = ',')
                reader = list(reader)
                r = [row[2:9] for row in reader]
                r = r[8:]
                r = np.array(r).astype(np.float64)
                
        return r
    
    #Opens intensity Data from csv
    def open_intensity_csv(self):
        files = tkFileDialog.askopenfilenames()
        if files:
            files = list(files)
            with open(files[0], 'rU') as csvfile:
                reader = csv.reader(csvfile, delimiter = ',')
                reader = list(reader)
                reader = np.array(reader)
                
        return reader
        
    #Gets indices of upward value swing
    def get_upswing(self, channel, threshold):
        return [index for index in range(1, len(channel)) if (channel[index] > threshold) and (channel[index - 1] < threshold)]
    
    #Gets indices of downward value swing
    def get_downswing(self, channel, threshold):
        return [index for index in range(1, len(channel)) if (channel[index] < threshold) and (channel[index - 1] > threshold)]
    
    #Gets cell data arrays from intesnity csv
    def get_cells(self, c):
        cells = list()
        areas = list()
        xcoords = list()
        ycoords = list()
        for i in range(len(c[0])):
            if 'RawIntDen' in c[0][i]:
                cells.append(np.array(c[1:, i]).astype(np.float64))
            if 'Area' in c[0][i]:
                areas.append(np.array(c[1, i]).astype(np.float64))
            if 'X' in c[0][i]:
                xcoords.append(np.array(c[1, i]).astype(np.float64))
            if 'Y' in c[0][i]:
                ycoords.append(np.array(c[1, i]).astype(np.float64))
        cells =  [np.divide(cells[i:], areas[i]) for i in range(len(areas))] # normalize IntDen to ROI area       
        return cells, areas, xcoords, ycoords
    
    def get_cells_from_smoothed(self, c):
        cells = list()
        for i in range(len(c[0])):
            cells.append(np.array(c[1:, i]).astype(np.float64))
        return cells
    
    #Gets channels from event csv 
    def get_channels(self, r):
        channels = [r[0:, i] for i in range(len(r[0]))]
        return channels
    
    #Gets event timestamp data from channels
    def get_event_stamps(self, channels, threshold):
        self.strobe_up = self.get_upswing(channels[1], threshold)
        self.strobe_down = self.get_downswing(channels[1], threshold)
        self.onset_up = self.get_upswing(channels[2], threshold)
        self.onset_down = self.get_downswing(channels[2], threshold)
        
        self.stim_up = self.get_upswing(channels[3], threshold)
        self.stim_down = self.get_downswing(channels[3], threshold)
        
        self.Bframe_up = self.get_upswing(channels[6], threshold)
        self.Bframe_down = self.get_downswing(channels[6], threshold)
    
    #Gets nearest value in array to input
    def get_nearest(self, x, array):
        keylist = np.array(array)
        index = (np.abs(keylist-x)).argmin()
        return keylist[index]
    
    #Gives maximum difference in ms bewteen frame onsets and stim onset
    def max_frame_error(self, frame_timestamps, event_timestamps):
        deviations = list()
        for ts in event_timestamps:
            dev = np.abs(np.array(frame_timestamps) - ts).min()
            deviations.append(dev)
        return deviations
    
    #Gives flip and flop dictionaries
    #key is (start, end) frames for that stimulus
    #value is intesnity data
    def get_flip_flops(self, cell, strobe_timestamps, stim_timestamps, frame_timestamps):
        duration = strobe_timestamps[1] - strobe_timestamps[0]
        flips = collections.OrderedDict()
        flops = collections.OrderedDict()
        gray = collections.OrderedDict()
        for ts in stim_timestamps:
            closest_flip = self.get_nearest(ts, strobe_timestamps)
            closest_flop = self.get_nearest(ts + duration, strobe_timestamps)
            
            flip_frame_ts = self.get_nearest(closest_flip, frame_timestamps)
            flop_frame_ts = self.get_nearest(closest_flop, frame_timestamps)
            
            flip_frame_ts_end = self.get_nearest(closest_flip + duration, frame_timestamps)
            flop_frame_ts_end = self.get_nearest(closest_flop + duration, frame_timestamps)
            gray_frame_ts_end = self.get_nearest(closest_flop + 2*duration, frame_timestamps)
            
            flip_frame = (np.abs(frame_timestamps - flip_frame_ts)).argmin()
            flop_frame = (np.abs(frame_timestamps - flop_frame_ts)).argmin()
            gray_frame = (np.abs(frame_timestamps - gray_frame_ts_end)).argmin()
            
            flip_frame_end = (np.abs(frame_timestamps - flip_frame_ts_end)).argmin()
            flop_frame_end = (np.abs(frame_timestamps - flop_frame_ts_end)).argmin()
            
            
            flips[(flip_frame, flip_frame_end)] = cell[flip_frame:flip_frame_end]
            flops[(flop_frame, flop_frame_end)] = cell[flop_frame: flop_frame_end]
            gray[(flop_frame_end, gray_frame)] = cell[flop_frame_end: gray_frame]
            
        return flips, flops, gray
    
    def get_stim_block(self, cell, stim_up, stim_down, frame_timestamps):
        stim_up.sort()
        stim_down.sort()
        stamps = zip(stim_up, stim_down)
        blocks = dict()
        for ts in stamps:
            onset_frame_ts = self.get_nearest(ts[0], frame_timestamps)
            offset_frame_ts = self.get_nearest(ts[1], frame_timestamps)
            
            frame_start = (np.abs(frame_timestamps - onset_frame_ts)).argmin()
            frame_end = (np.abs(frame_timestamps - offset_frame_ts)).argmin()
            
            blocks[(frame_start, frame_end)] = cell[frame_start:frame_end]
        return blocks
            
        
    #gets the average of flips or flops for a cell
    def get_avg(self, flips):
        return np.mean(np.array(flips).astype(np.float64), axis = 0)
    
    
def plotData(data, fig, span):
    plt.figure(fig)
    maximum = span[1]
    if span[1] > len(data):
        maximum = len(data)
    for i in range(span[0], maximum):
        plt.subplot(maximum-span[0], 1, i + 1)
        plt.plot(data[i])

    plt.show()

def calc_delta_f(cells):
    delta_f = list()
    for i in range(len(cells)):
        #frame_buffer = self.intensity_data[cell][0:frames]
        #f0 = np.average(np.array(frame_buffer).astype(np.float))
        f0 = float(min(cells[i]))
        delta = np.array(cells[i]).astype(np.float) - f0
        #if f0 > 1:
        delta /= f0
        delta_f.append(delta)
    return delta_f

    #calculates delta f based on mean gray (mean of gray-session means)
    #   ?Try 1s baseline window w/ gray and end of flip
def calc_delta_f_f0gray(cells, f0):
    delta = list()
    delta_f = list()
    for i in range(len(cells)):
        #frame_buffer = self.intensity_data[cell][0:frames]
        #f0 = np.average(np.array(frame_buffer).astype(np.float))
        #f0 = float(min(cells[i]))
        if isinstance(cells, dict):
            sessions = list()
            for j in range(len(cells[i])):
                sessions.append(np.array(cells[i].values()[j]).astype(np.float) - f0[i])
            delta.append(np.array(np.mean(sessions, axis=0)))
        elif isinstance(cells, list):
            delta = np.array(cells[i].values()[i]).astype(np.float) - f0[i]
        #if f0 > 1:
        delta /= f0[i]
        delta_f.append(np.asarray(delta))
        delta = list()
    return delta_f

def calc_stim_delta_f(flips, flops, gray, mode = 'mean'):
    delta_f_flips = dict()
    delta_f_flops = dict()
    delta_f_grays = dict()
    sorted_flips =  sorted(flips.keys(), key=lambda tup: tup[0])
    sorted_flops =  sorted(flops.keys(), key=lambda tup: tup[0])
    sorted_grays =  sorted(gray.keys(), key=lambda tup: tup[0])
    if mode == 'mean':
        f0 = np.mean(np.mean(np.array(gray.values()), axis = 0))
    if mode == 'min':
        f0 = np.min(np.mean(np.array(gray.values()), axis = 0))
    for i in range(len(sorted_flips)): # skip the session 1 (no gray)
        #print str(sorted_flips[i]) + " " + str(sorted_flops[i]) + " " + str(sorted_grays[i])
#        if mode == 'mean':
#            f0 = np.mean(gray[sorted_grays[i]])
#        elif mode == 'min':
#            f0 = np.min(gray[sorted_grays[i]])
        delta_f_flips[sorted_flips[i]] = (np.array(flips[sorted_flips[i]]).astype(np.float) - f0) / f0
        delta_f_flops[sorted_flops[i]] = (np.array(flops[sorted_flops[i]]).astype(np.float) - f0) / f0
        delta_f_grays[sorted_grays[i]] = (np.array(gray[sorted_grays[i]]).astype(np.float) - f0) / f0
        #move average to 0 and normalize
#        delta_f_flips[sorted_flips[i]] = delta_f_flips[sorted_flips[i]] - np.mean(delta_f_flips[sorted_flips[i]])
#        delta_f_flops[sorted_flops[i]] = delta_f_flops[sorted_flops[i]] - np.mean(delta_f_flops[sorted_flops[i]])
#        delta_f_grays[sorted_grays[i]] = delta_f_grays[sorted_grays[i]] - np.mean(delta_f_grays[sorted_grays[i]])
#        
#        delta_f_flips[sorted_flips[i]] = delta_f_flips[sorted_flips[i]] / np.max(delta_f_flips[sorted_flips[i]])
#        delta_f_flops[sorted_flops[i]] = delta_f_flops[sorted_flops[i]] / np.max(delta_f_flops[sorted_flops[i]])
#        delta_f_grays[sorted_grays[i]] = delta_f_grays[sorted_grays[i]] / np.max(delta_f_grays[sorted_grays[i]])
    return delta_f_flips, delta_f_flops, delta_f_grays
    
def calc_stim_delta_f2(flips, flops, grays, mode = 'mean'):
    delta_f_flips = collections.OrderedDict()
    delta_f_flops = collections.OrderedDict()
    delta_f_grays = collections.OrderedDict()
    for i in range(len(flips)): 
        #print str(sorted_flips[i]) + " " + str(sorted_flops[i]) + " " + str(sorted_grays[i])
        if mode == 'mean':
            temp1 = np.array([x[-30:] for x in grays[i].values()])
            f0 = np.mean([x for x in temp1], axis=1)
            """
            f0 is mean of last 30 frames of gray for 6 sessions
            Could make ’30’ an argument for function, make a catch if too long
            """
        elif mode == 'min':
            f0 = np.min(grays[sorted_grays[i]])
        # Loop through each cell by session (and only sessions 1,2,3,4,5 since 0 has no pre-gray)
        for i in range(len(flips)):
            zero1 = list()
            zero2 = list()
            zero3 = list()
            # Calc deltaf/f and zero each session-trace (maybe zero after concat?)
            for j in range(1,len(f0)): # skip session 1 (no preceding gray)
                dff1 = ( (np.array(flips[i].values()[j]).astype(np.float) - f0[j]) )/ f0[j]
                #min1 = dff1 - np.mean(dff1)
                #min1 -=min(min1)
                zero1.append(dff1)
                dff2 = ( (np.array(flops[i].values()[j]).astype(np.float) - f0[j]) )/ f0[j]
                zero2.append(dff2)
                dff3 = ( (np.array(grays[i].values()[j]).astype(np.float) - f0[j]) )/ f0[j]
                zero3.append(dff3)
            delta_f_flips[i] = ( zero1 )
            delta_f_flops[i] = ( zero2 )
            delta_f_grays[i] = ( zero3 )
    return delta_f_flips, delta_f_flops, delta_f_grays

def remove_block(stim, num):
    sorted_stims = sorted(stim.keys(), key = lambda tup: tup[0])
    del stim[sorted_stims[num]]
        

#test function for finding p-values
def get_cell_response(cell, window):
    flips, flops = Data.get_flip_flops(cell, Data.strobe_down, Data.stim_up, Data.Bframe_up)
    flip_vals = np.array(flips.values())
    flip_vals = flip_vals.transpose()
    for i in range(len(flip_vals)-window):
        groups = [flip_vals[j] for j in range(i, i + window)]
        if stats.f_oneway(*groups)[1] < 0.05:
            print "response detected"

def normalize_signals(delta_f):
    norms = list()
    for i in range(len(delta_f)):
        norms.append(delta_f[i] / np.max(delta_f[i]))
    return norms
        
def get_colormaps(delta_f, keys, minimum, maximum):
    delta_f = np.array(delta_f).astype(np.float)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(delta_f, vmin = minimum, vmax = maximum)
    ax.set_yticks(np.arange(len(delta_f))+0.5, minor=False)
    ax.set_yticklabels(keys, minor=False, fontsize = 8)
    ax.invert_yaxis()
    plt.colorbar(heatmap)
    plt.show()

def get_response_graph(delta_f, flips, flops, figure):
    plt.figure(figure)
    for i in range(len(delta_f)):
        plt.plot(delta_f[i] + float(i))
    plt.axis('tight')
    for ts in flips[0].keys():
        plt.axvline(x = ts[0], color = 'red')
        plt.axvline(x = ts[1], color = 'green', ls = 'dashed')
    for ts in flops[0].keys():
        plt.axvline(x = ts[1], color = 'black')

def get_prestim_values(cells, stim, i):
    values = dict()
    for ts in stim[i]:
        values[(ts[0] - (ts[1] - ts[0]), ts[0])] = cells[i][ts[0] - (ts[1] - ts[0]): ts[0]]
    return values

def is_responsive(stim_stats, cells):
    responsive = list()
    nonresponsive = list()
    for i in stim_stats:
        if stim_stats[i][1] < 0.01:
            #print str(i) + " responsive"
            responsive.append(i)
        elif stim_stats[i][1] >= 0.01:
            nonresponsive.append(i)
    return responsive, nonresponsive

def is_thresh_responsive(stim, cells):
    pass

def add_stim_bars(flips, flops, fig):
    plt.figure(fig)
    plt.axis('tight')
    for ts in flips[0].keys():
        plt.axvline(x = ts[0], color = 'red')
        plt.axvline(x = ts[1], color = 'green', ls = 'dashed')
    
    for ts in flops[0].keys():
        plt.axvline(x = ts[1], color = 'teal')

def topfig(userY): #added by JEC - arrange figs on window
    figmgr = get_current_fig_manager()
    figmgr.canvas.manager.window.raise_()
    geom = figmgr.window.geometry()
    x,y,dx,dy = geom.getRect()
    figmgr.window.setGeometry(10, userY, dx, dy)
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_flip_flop_gray(data, Data): # data = delta_f 45 or 135; need stim offsetif filtered???      
    #Get flip flops
    flips = collections.OrderedDict()
    flops = collections.OrderedDict()
    grays = collections.OrderedDict()
    
    for i in range(len(data)):
        flips[i], flops[i], grays[i] = Data.get_flip_flops(data[i], Data.strobe_down, Data.stim_up, Data.Bframe_up)
    
    #chop flips and flops down to min length
    for arr in flips:
        min_len = min([len(i) for i in flips[arr].values()])
        for ts in flips[arr]:
            flips[arr][ts] = flips[arr][ts][0:min_len]
    
    for arr in flops:
        min_len = min([len(i) for i in flops[arr].values()])
        for ts in flops[arr]:
            flops[arr][ts] = flops[arr][ts][0:min_len]
    
    for arr in grays:
        min_len = min([len(i) for i in grays[arr].values()])
        for ts in grays[arr]:
            grays[arr][ts] = grays[arr][ts][0:min_len]           
    return flips, flops, grays

def classify_responses(data, flips, flops, grays):
    toggleClassifyCells = 'False' # parse data into "responsive (r) or non-responseive (nr)" categories   
    if toggleClassifyCells == 'True':
    
        flip_stats = dict()
        flop_stats = dict()
        gray_stats = dict()
        flip_prestims = dict()
        flop_prestims = dict()
        
        flip_thresh = dict()
        flop_thresh = dict()
        gray_thresh = dict()
        
        #frames to remove fron end in ANOVA
        frames = 1
            
        #delete bad ROI
        if filetype_data == 2:
            del flips[168]
            del flops[168]
            del grays[168]
        
    
        
        for i in range(len(flips)):
            #get rid of bad first array
            remove_block(flips[i], 0)
            remove_block(flops[i], 0)
            remove_block(grays[i], 0)
    #        remove_block(flips[i], 0)
    #        remove_block(flops[i], 0)
    #        remove_block(flips[i], 3)
    #        remove_block(flops[i], 3)
            
            flip_prestims[i] = get_prestim_values(data, flips, i)
            flop_prestims[i] = get_prestim_values(data, flops, i)
            
            
            #remove end frames
            flipv = np.array(flips[i].values())[:-frames]
            flopv = np.array(flops[i].values())[:-frames]
            grayv = np.array(grays[i].values())[:-frames]
    #        flip_prev = np.array(flip_prestims[i].values())[:,frames:]
    #        flop_prev = np.array(flop_prestims[i].values())[:,frames:]
            
    #        flip_vals = np.concatenate((flipv, flip_prev), axis = 0)
    #        flop_vals = np.concatenate((flopv, flop_prev), axis = 0)
            
            thr = 1.0
            flip_truths = [f > thr for f in flipv]
            flip_thresh[i] = [True in X for X in flip_truths]
            flop_truths = [f > thr for f in flopv]
            flop_thresh[i] = [True in X for X in flop_truths]
            gray_truths = [f > thr for f in grayv]
            gray_thresh[i] = [True in X for X in gray_truths]
            
            #test case
            flip_vals = flipv
            flop_vals = flopv
            gray_vals = grayv
               
            #flop_vals = np.array(np.concatenate((flops[i].values(), flop_prestims[i].values()), axis = 0)).astype(np.float64)
            #flip_vals = np.array(np.concatenate((flips[i].values(), flip_prestims[i].values()), axis = 0)).astype(np.float64)
            
            flip_vals = flip_vals.transpose()
            flop_vals = flop_vals.transpose()
            gray_vals = gray_vals.transpose()
            
            flip_stats[i] = stats.kruskal(*flip_vals)
            flop_stats[i] = stats.kruskal(*flop_vals)
            gray_stats[i] = stats.kruskal(*gray_vals)
    

        #Graphing Procedure
        for i in range(len(data)):
            cells[i] = np.nan_to_num(data[i])
        
        #FLIPS
        r_flip_keys, nr_flip_keys = is_responsive(flip_stats, data)
        r_flips = [data[i] for i in r_flip_keys]
        nr_flips = [data[i] for i in nr_flip_keys]
        
        r_flip_thresh_keys = [i for i in flip_thresh if True in flip_thresh[i]]
        r_flop_thresh_keys = [i for i in flop_thresh if True in flop_thresh[i]]
        r_gray_thresh_keys = [i for i in gray_thresh if True in gray_thresh[i]]
        
        max_val = max([np.max(i) for i in r_flips])
        min_val = min([np.min(i) for i in r_flips])
        
        #responsive flips graph
    #    r_flip_norms = sorted(normalize_signals(r_flips), key = lambda x: np.argmax(x))
    #    r_flip_norms.reverse()
    #    get_colormaps(r_flip_norms, min_val, max_val, 6)
    #    add_stim_bars(flips, flops, 6)
    #    
    #    #nonresponsive flips graph
    #    nr_flip_norms = sorted(normalize_signals(nr_flips), key = lambda x: np.argmax(x))
    #    nr_flip_norms.reverse()
    #    get_colormaps(nr_flip_norms, min_val, max_val, 7)
    #    add_stim_bars(flips, flops, 7)
        
        #FLOPS
        r_flop_keys, nr_flop_keys = is_responsive(flop_stats, data)
        r_flops = [data[i] for i in r_flop_keys]
        nr_flops = [data[i] for i in nr_flop_keys]
        
    #    max_flop_response = max([np.max(i) for i in r_flops])
    #    min_flop_response = min([np.min(i) for i in r_flops])
    #    
    #    #responsive flops graph
    #    r_flop_norms = sorted(r_flops, key = lambda x: np.argmax(x))
    #    r_flop_norms.reverse()
    #    get_colormaps(r_flop_norms, min_val, max_val, 8)
    #    add_stim_bars(flips, flops, 8)
    #    
    #    #nonresponsive flop norms
    #    nr_flop_norms = sorted(nr_flops, key = lambda x: np.argmax(x))
    #    nr_flop_norms.reverse()
    #    get_colormaps(nr_flop_norms,min_val, max_val, 9)
    #    add_stim_bars(flips, flops, 9)
        
        r_gray_keys, nr_gray_keys = is_responsive(gray_stats, data)
        r_grays = [data[i] for i in r_gray_keys]
        nr_grays =[data[i] for i in nr_gray_keys]
        
        #BOTH
        r_both_keys = list(set(r_flip_keys) | set(r_flop_keys) | set(r_gray_keys))
        nr_both_keys = list(set(nr_flip_keys) & set(nr_flop_keys) & set(nr_gray_keys))
        r_both = [data[i] for i in r_both_keys]
        nr_both = [data[i] for i in nr_both_keys]
        
        #FLIP AVGS
        r_flip_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in r_flip_keys}
        nr_flip_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in nr_flip_keys}
        r_flop_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in r_flop_keys}
        nr_flop_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in nr_flop_keys}
        r_gray_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in r_gray_keys}
        nr_gray_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in nr_gray_keys}
        
    
        max_flip_val = max(max([np.max(i) for i in r_flip_avgs]), max([np.max(i) for i in nr_flip_avgs]))
        max_flop_val = max(max([np.max(i) for i in r_flop_avgs]),max([np.max(i) for i in nr_flop_avgs]))
        min_flip_val = min(min([np.min(i) for i in r_flip_avgs]),min([np.min(i) for i in nr_flip_avgs]))
        min_flop_val = min(min([np.min(i) for i in r_flop_avgs]),min([np.min(i) for i in nr_flop_avgs]))
    
        #BOTH AVGS
        r_flip_both_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in r_both_keys}
        nr_flip_both_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in nr_both_keys}
        r_flop_both_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in r_both_keys}
        nr_flop_both_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in nr_both_keys}
        r_gray_both_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in r_both_keys}
        nr_gray_both_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in nr_both_keys}        
        
        r_both_avgs = {i:np.concatenate((r_flip_both_avgs[i], r_flop_both_avgs[i], r_gray_both_avgs[i]), axis = 0) for i in r_both_keys}
        nr_both_avgs = {i:np.concatenate((nr_flip_both_avgs[i], nr_flop_both_avgs[i], nr_gray_both_avgs[i]), axis = 0) for i in nr_both_keys}
    #    r_avgs_sorted = sorted(r_both_avgs, key = lambda x: np.argmax(x))
    #    r_avgs_sorted.reverse()        
        return r_both_avgs, nr_both_avgs,         
        # END CLASSIFY RESPONSES

def avgSort_FlipFlopGray(flips, flops, grays):    
    all_flip_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in flips.keys()}
    all_flop_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in flops.keys()}
    all_gray_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in grays.keys()}        
    all_avgs = {i:np.concatenate((all_flip_avgs[i], all_flop_avgs[i], all_gray_avgs[i]), axis = 0) for i in all_flip_avgs.keys()}
    
    #Zero all data (data-mean(data))
    for key in all_avgs:
        all_avgs[key] -= np.mean(all_avgs[key])
        all_avgs[key] -= np.min(all_avgs[key])
        all_avgs[key] = np.array(all_avgs[key]).astype(np.float) 

    #Normalize all data (0-1)    
    norm_all_avgs = copy.deepcopy(all_avgs) #.copy()
    for key in norm_all_avgs:
        norm_all_avgs[key] -= min(norm_all_avgs[key])
        norm_all_avgs[key] /= max(norm_all_avgs[key])
        norm_all_avgs[key] = np.array(norm_all_avgs[key]).astype(np.float)        

    all_avgs_sorted_keys = sorted(all_avgs, key = lambda key: np.argmax(norm_all_avgs[key]))
    all_avgs_sorted = list()
    for key in all_avgs_sorted_keys:
        all_avgs_sorted.append(all_avgs[key])
        
    norm_avgs_sorted_keys = sorted(norm_all_avgs, key = lambda key: np.argmax(norm_all_avgs[key]))
    norm_avgs_sorted = list()
    for key in norm_avgs_sorted_keys:
        norm_avgs_sorted.append(norm_all_avgs[key])
       
    return all_flip_avgs, all_flop_avgs, all_gray_avgs, all_avgs, norm_all_avgs, all_avgs_sorted_keys, norm_avgs_sorted_keys, all_avgs_sorted, norm_avgs_sorted

def plotHeatAvgs(data, datakeys, timestamps_flip, timestamps_flop, minheat, maxheat, plottitle):
    get_colormaps(data, datakeys, minheat, maxheat)
    plt.title(plottitle)
    plt.axvline(x = len(timestamps_flip[0].values()[0]), color = 'green', ls = 'dashed')
    plt.axvline(x = len(timestamps_flip[0].values()[0]) + len(timestamps_flop[0].values()[0]), color = 'green', ls = 'dashed')
    plt.axis('tight')
    #plt.savefig('All_Cell_Averages_(ANOVA).eps', format = 'eps', dpi = 1200)
    #topfig(800)
    
def figSlowStimTrace(data,step_factor):
    fig, ax = plt.subplots()
    for k in range(len(data)):
        plt.plot(data[k][:]+k*step_factor)
        
def figSlowStimTrace2(data,step_factor):
    fig, ax = plt.subplots()
    for k in range(len(data)):
        plt.plot(data[k][0]+k*step_factor)
    
def plotSessionTraces(nrows, ncols, sessiondata, avgdata, *args):
    fig, axes = plt.subplots(nrows, ncols, sharex='all', sharey='all')      
      
    for i in range(nrows):
        for j in range(ncols):
            for k in range(len(sessiondata[i*ncols+j])):
                axes[i,j].plot(sessiondata[i*ncols+j][k])
                #temptitle = 'Cell '+str(i*ncols+j)
                #axes[i,j].title(temptitle)
                
            axes[i,j].plot(avgdata[i*ncols+j], linewidth=2, color='k')
            axes[i,j].text(.5, .8,'Cell '+str(i*ncols+j+1), ha='left', va='center', transform=axes[i,j].transAxes)
            if i==0 and j==0:
                axes[i,j].tick_params(axis=u'both', which=u'both',length=0)
            else:
                axes[i,j].axis('off')
            if len(args)>0: # value for timestamp
                axes[i,j].axvline(x=args[0], ymin=0, ymax = len(avgdata), linewidth=0.5, color='k')
    
#def EWMA(z,L):
#    #    %
#    #    % Computes the exponentially weighted moving average (with memory L) of
#    #    % input data z
#    #    %
# MATLAB:   
#    lambda = 1-2/(L+1);
#    
#    smoothed_z = zeros(size(z));
#    for i = 1:size(z,1)
#        smoothed_z(i,1) = z(i,1);
#        for j = 2:size(z,2)
#            smoothed_z(i,j) = lambda * smoothed_z(i,j-1) + (1-lambda) * z(i,j);
#        end
#    end
#    return smoothed_z

#def process_function(signalin, t_0, t_1, t_2, samplingfreq)
#%Implementation of Nature protocol
#%Hongbo Jia, Nathalie L Rochefort1, Xiaowei Chen & Arthur Konnerth1 "In
#%vivo two-photon imaging of sensory-evoked dendritic calcium signals in cortical neurons"
#%
#%Implementation copyright Petros Xanthopoulos 2013-2014
#%usage: signalout=process_function(signalin,t_0,t_1,t_2)
#% where
#% input: signalin is the raw signal 
#%t_0,t_1,t_2 are the parameters described in Nature protocol paper
#%comments: for a 30Hz imaging systems the following parameter setup is
#%recommended (empirical note on Nature paper): 
#%t_0= 0.2;
#%t_1=0.75;
#%t_2=3;
#
#
#function signalout=process_function_jc(signalin,t_0,t_1,t_2,samplingfreq)
#
#F_0=[];
#
#Fs=samplingfreq; %sampling frequency
#
#t_0_s=floor(t_0*Fs);
#t_1_s=floor(t_1*Fs);
#t_2_s=floor(t_2*Fs);
#
#F_sm = smooth(signalin,t_1_s);
#
#for i=(t_2_s+1):length(signalin)
#    F_0=[F_0 min(F_sm(i-t_2_s:i))];
#end
#
#R_0=(F_sm((t_2_s+1):end)-F_0')./F_0';
#
#R_0_sm = EWMA(R_0,t_0_s);
#
#
#signalout=R_0_sm;   
#
# return signalout         


if __name__ == "__main__":
    
    filetype_data = 1    # 1 = group _001 cells; 2 = _002 cells (169 ROIs)
    filetype_csv = 3 # 1 = EWMA dff CSV; 2 = raw intensity CSV; 3 = FIJI ROI CSV
    
    ########################
    
    root = tk.Tk()
    threshold = 2.5
    root.update()
    
    Data = FileHandler()
    Data45 = FileHandler()
    Data135 = FileHandler()    
    
    if filetype_csv == 1:
        int_smooth_red = Data.open_intensity_csv()
        cells = Data.get_cells_from_smoothed(int_smooth_red)
    elif filetype_csv == 2:
        int_raw_red = Data.open_intensity_csv()
        cells = Data.get_cells_from_smoothed(int_raw_red)
        delta_f = calc_delta_f(cells)
    elif filetype_csv == 3:
        intensity_data = Data.open_intensity_csv()
        csvdata = Data.get_cells(intensity_data)

        allcells = csvdata[0][0]
        areas = csvdata[1]
        xycoords = [csvdata[2], csvdata[3]] #x=xycoords[0][0], y = xycoords[1][0]
        
        if len(allcells[0])==7200:
            print('*** Select concat stack (45-135) ***')
            cells45 = allcells[:,0:3600]
            cells135 = allcells[:,3600:7200]    

        run_lopass = 1
        # Run Low-pass filter - USAGE: (cells = raw intensity data)
        if run_lopass == 1:
            cells45_lopass = list()
            cells135_lopass = list()
            cutoff = 0.8 # in Hz Rose et al. used 0.8Hz
            fs = 30 # sampling freq in Hz
            order = 2 # not sure what this should be exactly?
            for i in range(len(cells45)):
                data = cells45[i][:]
                y = butter_lowpass_filter(data, cutoff, fs, order)
                cells45_lopass.append(y)
            data = None
            for i in range(len(cells135)):
                data = cells135[i][:]
                y = butter_lowpass_filter(data, cutoff, fs, order)
                cells135_lopass.append(y)
            data = None
        
#    deltaff_45 = calc_delta_f(cells45_lopass)
#    deltaff_135 = calc_delta_f(cells135_lopass)

    print('*** Get timestamps for T5 45deg ***')    
    event_data45 = Data45.open_event_csv()
    channels45 = Data45.get_channels(event_data45)    
    Data45.get_event_stamps(channels45, threshold)    
    # Get stim and gray chunks
    flips45, flops45, grays45 = get_flip_flop_gray(cells45_lopass, Data45)
    
    print('*** Get timestamps for T5 135deg ***')
    event_data135 = Data135.open_event_csv()
    channels135 = Data135.get_channels(event_data135)    
    Data135.get_event_stamps(channels135, threshold)
    # Get stim and gray chunks
    flips135, flops135, grays135 = get_flip_flop_gray(cells135_lopass, Data135)
    
    #delta_f_flips45, delta_f_flops45, delta_f_grays45 = calc_stim_delta_f(flips45, flops45, gray45, mode = 'mean')    
    
    
    """
    # Pull out gray average for f0 in calc_delta_f()
    temp = list()
    grays45_mean = list()
    for key in grays45:
        for i in range(len(grays45[key].values())):
            temp.append(np.mean(grays45[key].values()[i]))
        grays45_mean.append(np.mean(temp))
        temp = list()
    temp = list()
    grays135_mean = list()
    for key in grays45:
        for i in range(len(grays135[key].values())):
            temp.append(np.mean(grays135[key].values()[i]))
        grays135_mean.append(np.mean(temp))
        temp = list()       
    deltaf_flips45 = calc_delta_f_f0gray(flips45, grays45_mean) # data, f0 (list)
    deltaf_flops45 = calc_delta_f_f0gray(flops45, grays45_mean) # data, f0 (list)
    deltaf_grays45 = calc_delta_f_f0gray(grays45, grays45_mean) # data, f0 (list)
    
    deltaf_flips135 = calc_delta_f_f0gray(flips135, grays135_mean)
    deltaf_flops135 = calc_delta_f_f0gray(flops135, grays135_mean)
    deltaf_grays135 = calc_delta_f_f0gray(grays135, grays135_mean)
    """
    
    root.mainloop()

#"""    

    # Sorting and plotting all averages (flip-flop-gray avgs)    
    all45_flip_avgs, all45_flop_avgs, all45_gray_avgs, all45_avgs, norm45_all_avgs, all45_avgs_sorted_keys, norm45_avgs_sorted_keys, all45_avgs_sorted, norm45_avgs_sorted = avgSort_FlipFlopGray(flips45, flops45, grays45)
    all135_flip_avgs, all135_flop_avgs, all135_gray_avgs, all135_avgs, norm135_all_avgs, all135_avgs_sorted_keys, norm135_avgs_sorted_keys, all135_avgs_sorted, norm135_avgs_sorted = avgSort_FlipFlopGray(flips135, flops135, grays135)


    max(map(max, all45_avgs_sorted))
    max(map(max, all135_avgs_sorted))
    plotHeatAvgs(all45_avgs_sorted, all45_avgs_sorted_keys, flips45, flops45, 0,200.0, 'T5 001 45deg - zero')
    plotHeatAvgs(all135_avgs_sorted, all135_avgs_sorted_keys, flips135, flops135, 0,200.0, 'T5 001 135deg - zero')

    
    #Zero data
    #Zero all data (data-mean(data))
    for key in all135_avgs:
        all135_avgs[key] -= np.mean(all135_avgs[key])
        all135_avgs[key] = np.array(all135_avgs[key]).astype(np.float)
        
        
    # Sort 135 data based on 45 data
    temp135_avgs_sort45 = list()
    for key in all45_avgs_sorted_keys:
        temp135_avgs_sort45.append(all135_avgs[key])
        
        
    get_colormaps(temp135_avgs_sort45, all45_avgs_sorted_keys, 0, 30.0)
    plt.title('T5 001 135 deg 45-sort')
    plt.axvline(x = len(flips45[0].values()[0]), color = 'green', ls = 'dashed')
    plt.axvline(x = len(flips45[0].values()[0]) + len(flops45[0].values()[0]), color = 'green', ls = 'dashed')
    plt.axis('tight')
    
    figSlowStimTrace(temp135_avgs_sort45,2)
    
    min(map(min, all45_avgs_sorted))
    

    threshold = 10 # for f0=min
    data = all45_avgs
    datakeys = all45_avgs_sorted_keys
    
    # def plotHeat_TH(threshold, data, datakeys)
    newdata=list()
    newdatakeys=list()
    #axhline(y=0.2,linewidth=1, color='r')
    for i in range(len(datakeys)):   
        tempkey = datakeys[i]
        if max(data[tempkey]) > threshold:
            newdata.append(data[tempkey])
            newdatakeys.append(tempkey)
            
    get_colormaps(newdata, newdatakeys, 0, 30.0)
    plt.title('T5 001 45deg - norm TH')
    plt.axvline(x = len(flips135[0].values()[0]), color = 'green', ls = 'dashed')
    plt.axvline(x = len(flips135[0].values()[0]) + len(flops135[0].values()[0]), color = 'green', ls = 'dashed')
    plt.axis('tight')
            
    

    
#"""




"""
#Sorting and plotting r and nr averages (flip-pflop-gray avgs)
      
r_gray_avg_sorted_keys = sorted(r_gray_avgs, key = lambda key: np.argmax(r_gray_avgs[key]))
nr_gray_avg_sorted_keys = sorted(nr_gray_avgs, key = lambda key: np.argmax(nr_gray_avgs[key]))


r_avgs_sorted_keys = sorted(r_both_avgs, key = lambda key: np.argmax(r_both_avgs[key]))
r_avgs_sorted = list()
for key in r_avgs_sorted_keys:
    r_avgs_sorted.append(r_both_avgs[key])
get_colormaps(r_avgs_sorted,r_avgs_sorted_keys, 0, 0.6)
plt.title('Responsive Cell Averages (ANOVA)').set_family('helvetica')
plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]) + len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
plt.axis('tight')
plt.savefig('Responsive_Cell_Averages_(ANOVA).eps', format = 'eps', dpi = 1200)
#topfig(10)

#    nr_avgs_sorted = sorted(nr_both_avgs, key = lambda x: np.argmax(x))
#    nr_avgs_sorted.reverse()
nr_avgs_sorted_keys = sorted(nr_both_avgs, key = lambda key: np.argmax(nr_both_avgs[key]))
nr_avgs_sorted = list()
for key in nr_avgs_sorted_keys:
    nr_avgs_sorted.append(nr_both_avgs[key])
get_colormaps(nr_avgs_sorted, nr_avgs_sorted_keys, 0, 0.6)
plt.title('Non-Responsive Flip / Flop Averages (ANOVA)')
plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]) + len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
plt.axis('tight')
plt.savefig('Non-responsive_Cell_Averages_(ANOVA).eps', format = 'eps', dpi = 1200)
#topfig(600)
"""    
    


    