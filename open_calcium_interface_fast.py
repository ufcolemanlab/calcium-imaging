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
from scipy import signal
from scipy.signal import butter, lfilter, freqz

plt.rcParams["font.family"] = "helvetica"

class Cell():
    def __init__(self, name, intensity_data):
        self.name = name
        self.responsive = False
        
        self.intensity = intensity_data
        
        self.flips = dict()
        self.flops = dict()
        self.gray = dict()
        
        self.blocks = dict()

class SRPdecoder:
    def __init__(self, *args, **kwargs):

        self.voltageThreshold = 2.5
        self.stimLength = 500
        self.baseline = 25


    #Gives back chunks of the input signalChannel corresponding to flip and flop stim lists
    def GetStimLists(self, signalChannel, numStims, avgLength, flipTimeStamps, flopTimeStamps, frame_timestamps):
        frame_timestamps = np.array(frame_timestamps).astype(np.float)
        flips = dict()
        flops = dict()
        grays = dict()
        sessions = dict()
        block = 0
        
        last_flop_frame_end = None
                    
        for i in range(len(flopTimeStamps)):

            if self.StimLength(flopTimeStamps[i]) < 0.5*avgLength:
                first_flip_frame = (np.abs(frame_timestamps - flipTimeStamps[i][0])).argmin()
                current_flip_block = dict()
                current_flop_block = dict()
                current_gray_block = dict()

                if last_flop_frame_end is not None:                 
                    #first_flip_frame = (np.abs(frame_timestamps - flipTimeStamps[i][0])).argmin()
                    current_gray_block[(last_flop_frame_end, first_flip_frame)] = signalChannel[last_flop_frame_end:first_flip_frame]
                    grays[block - 1] = current_gray_block.copy()
                for j in range(numStims):
                    flip_frame = (np.abs(frame_timestamps - flipTimeStamps[i+j][0])).argmin()
                    flip_frame_end = (np.abs(frame_timestamps - (flipTimeStamps[i+j][0] + self.stimLength))).argmin()
                    current_flip_block[(flip_frame, flip_frame_end)] = (signalChannel[flip_frame:flip_frame_end])
                    flips[block] = current_flip_block.copy()
                for j in range(numStims - 1):
                    flop_frame = (np.abs(frame_timestamps - flopTimeStamps[1+i+j][0])).argmin()
                    flop_frame_end = (np.abs(frame_timestamps - (flopTimeStamps[1+i+j][0] + self.stimLength))).argmin()
                    #gray_frame_end = (np.abs(frame_timestamps - (flopTimeStamps[1+i+j][0] + 2 * self.stimLength))).argmin()
                    current_flop_block[(flop_frame, flop_frame_end)] = (signalChannel[flop_frame:flop_frame_end])
                    #current_gray_block[(flop_frame_end, gray_frame_end)] = signalChannel[flop_frame_end:gray_frame_end]
                #split the last long block into flip/flop
                last_flop_frame = (np.abs(frame_timestamps - (flipTimeStamps[i + numStims - 1][0] + self.stimLength))).argmin()
                last_flop_frame_end = (np.abs(frame_timestamps - (flipTimeStamps[i + numStims - 1][0] + 2 * self.stimLength))).argmin()
                #last_gray_frame_end = (np.abs(frame_timestamps - (flipTimeStamps[i + numStims - 1][0] + 3 * self.stimLength))).argmin()
                current_flop_block[(last_flop_frame, last_flop_frame_end)] = (signalChannel[last_flop_frame:last_flop_frame_end])
                #current_gray_block[(last_flop_frame_end, last_gray_frame_end)] = signalChannel[last_flop_frame_end:last_gray_frame_end]
                
                flops[block] = current_flop_block.copy()
                
                sessions[(first_flip_frame, last_flop_frame_end)] = signalChannel[first_flip_frame:last_flop_frame_end]
                
                block = block + 1
                current_flip_block.clear()
                current_flop_block.clear()
                current_gray_block.clear()
                
        gray_length = grays[1].keys()[0][0] - grays[0].keys()[0][1]
        current_gray_block[(last_flop_frame_end, last_flop_frame_end + gray_length)] = signalChannel[last_flop_frame_end:last_flop_frame_end + gray_length]
        grays[block - 1] = current_gray_block.copy()
                
                
        return flips, flops, grays, sessions
    
    #Averages flips/flops in sequential groups of numStims
    def GetAverages(self, flips, numStims):
        avgs = []
        for i in range(0, len(flips), numStims):
            avg = np.zeros(len(flips[0]))
            for j in range(numStims):
                avg += np.array(flips[i+j]-np.average(flips[i+j][:self.baseline]))
            avg /= numStims
            #avg += i*np.ones(len(flips[0]))
            avgs.append(avg)
            
        return avgs
            
                
    #Detects the number of stims in a session
    def StimsPerSession(self, flipLengths, flopLengths, avgLength):
        start = 0
        end = 0
        while(flopLengths[start] > avgLength*0.5):
            start += 1
        while(flipLengths[end] < avgLength*1.5):
            end += 1
        return end - start + 1
        
    #Make this take and return list of lists for using more than 2 channels (maybe a dictionary)
    def GetStimLengths(self, flipTimeStamps, flopTimeStamps):
        flipLengths = [self.StimLength(flip) for flip in flipTimeStamps]
        flopLengths = [self.StimLength(flop) for flop in flopTimeStamps]
        return flipLengths, flopLengths
        
    #takes a list of encoding channels and gives the code at sample point i
    def Decode(self, i, channels):
        return sum(2**a for (a,j) in enumerate([channel[i] for channel in channels]) if j > self.voltageThreshold)
    
    #Returns a code list corresponding to the sample in the signal
    #passes signal for standardization of channel length
    def GetCodeList(self, sig, channels):
        return[self.Decode(i, channels) for i in range(len(sig))]
            
    #Gives total number of sessions
    def GetTotalSessions(self, flopTimeStamps):
        return sum(self.StimLength(i) < 0.5*self.AvgStimLength([flopTimeStamps]) for i in flopTimeStamps)
    
    #Gives (rise, fall) timestamps corresponding to given code (1 for flop, 2 for flip)
    def GetTimeStamps(self, code, timeCodes):    
        rise = [i for i in range (1, len(timeCodes)) if timeCodes[i] == code and timeCodes[i-1] != code]
        fall = [i for i in range (1, len(timeCodes)) if timeCodes[i] != code and timeCodes[i-1] == code]
        return zip(rise, fall)
        
    #gives average stim length. input must be in a list, i.e. f([flipTimeStamps]) or f([flipTimes, flopTimes]) but not f(flipTimeStamps)
    def AvgStimLength(self, timeStampsLists):
        return sum(sum(y-x for (x,y) in stampsList)/len(stampsList) for stampsList in timeStampsLists)/len(timeStampsLists)
        
    def StimLength(self, stim):
        return stim[1]- stim[0]
        

        
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
            with open(files[0], 'rb') as csvfile:
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
        for i in range(len(c[0])):
            if 'RawIntDen' in c[0][i]:
                cells.append(np.array(c[1:, i]).astype(np.float64))
        return cells
    
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
    #value is intensity data
    def get_flip_flops(self, cell, strobe_timestamps, stim_timestamps, frame_timestamps):
        #cell, Data.strobe_down, Data.stim_up, Data.Bframe_up
        duration = strobe_timestamps[1] - strobe_timestamps[0]
        flips = dict()
        flops = dict()
        gray = dict()
        for ts in stim_timestamps:
            closest_flip = self.get_nearest(ts, strobe_timestamps)
            closest_flop = self.get_nearest(ts + duration, strobe_timestamps)
            
            flip_frame_ts = self.get_nearest(closest_flip, frame_timestamps)
            flop_frame_ts = self.get_nearest(closest_flop, frame_timestamps)
            
            flip_frame_ts_end = self.get_nearest(closest_flip + duration, frame_timestamps)
            flop_frame_ts_end = self.get_nearest(closest_flop + duration, frame_timestamps)
            gray_frame_ts_end = self.get_nearest(closest_flop + 2*duration, frame_timestamps)
            
            #frames are counted by how many times Bframe channel pulses up and down
            #target frame number should be the array index for the nearest timestamp
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

    #calculates delta f based on min fluorescence
    #Try 1s baseline window w/ gray and end of flip
def calc_delta_f(cells):
    delta_f = list()
    for i in range(len(cells)):
        #frame_buffer = self.intensity_data[cell][0:frames]
        #f0 = np.average(np.array(frame_buffer).astype(np.float))
        f0 = float(np.mean(cells[i]))
        delta = np.array(cells[i]).astype(np.float) - f0
        if f0 > 1:
            delta /= f0
        delta_f.append(delta)
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
    for i in range(len(sorted_flips)):
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
        if stim_stats[i][1] < 0.05:
            #print str(i) + " responsive"
            responsive.append(i)
        elif stim_stats[i][1] >= 0.05:
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
        
def Decode(i, channels, thresh):
    return sum(2**a for (a,j) in enumerate([channel[i] for channel in channels]) if j > thresh)
    
#Returns a code list corresponding to the sample in the signal
def GetCodeList(sig, channels, thresh):
    return[Decode(i, channels, thresh) for i in range(len(sig))]
    
    #Gives (rise, fall) timestamps corresponding to given code (1 for flop, 2 for flip)
def GetTimeStamps(code, timeCodes):    
    rise = [i for i in range (1, len(timeCodes)) if timeCodes[i] == code and timeCodes[i-1] != code]
    fall = [i for i in range (1, len(timeCodes)) if timeCodes[i] != code and timeCodes[i-1] == code]
    return zip(rise, fall)

def get_upswing(channel, threshold):
    return [index for index in range(1, len(channel)) if (channel[index] > threshold) and (channel[index - 1] < threshold)]

#Gets indices of downward value swing
def get_downswing(channel, threshold):
    return [index for index in range(1, len(channel)) if (channel[index] < threshold) and (channel[index - 1] > threshold)]

def GetStimLists(self, signalChannel, numStims, avgLength, flipTimeStamps, flopTimeStamps):
    flips = []
    flops = []
    for i in range(len(flopTimeStamps)):
        if self.StimLength(flopTimeStamps[i]) < 0.5*avgLength:
            for j in range(numStims):
                flips.append(signalChannel[flipTimeStamps[i+j][0]:flipTimeStamps[i+j][0]+self.stimLength])
            for j in range(numStims - 1):
                flops.append(signalChannel[flopTimeStamps[1+i+j][0]:flopTimeStamps[1+i+j][0] + self.stimLength])
            flops.append(signalChannel[flipTimeStamps[i + numStims - 1][0] + self.stimLength:flipTimeStamps[i + numStims-1][0]+2* self.stimLength])
            
    return flips, flops

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":
    root = tk.Tk()
    threshold = 2.5
    root.update()
    
    encoding = 'fast'
    
    ############################
    filetype = 1 # 1=rows-columns csv file; 2 = FIJI ROI csv file    
    
    Data = FileHandler()
    if filetype == 1:
        int_smooth_red = Data.open_intensity_csv()
        cells = Data.get_cells_from_smoothed(int_smooth_red)
    elif filetype == 2:
        intensity_data = Data.open_intensity_csv()
        cells = Data.get_cells(intensity_data)
        #delta_f = calc_delta_f(cells)

    timestamp_file = 1
    if timestamp_file == 1:
        event_data = Data.open_event_csv()
        channels = Data.get_channels(event_data)
        Data.get_event_stamps(channels, threshold)
    #elif timestamp_file == 2:
        # place new class/functions for extracting "voltage" timestamps
    ############################
    flips = dict()
    flops = dict()
    grays = dict()
    
    sessions = dict()
    
    delta_f_flips = dict()
    delta_f_flops = dict()
    delta_f_grays = dict()
    
    # USAGE: (cells = raw intensity data)
    cells_lopass = list()
    cutoff = 0.8 # in Hz Rose et al. used 0.8Hz
    fs = 30 # sampling freq in Hz
    order = 2 # not sure what this should be exactly?
    for i in range(len(cells)):
        data = cells[i][:]
        y = butter_lowpass_filter(data, cutoff, fs, order)
        cells_lopass.append(y)
    
    if encoding == 'slow':
        for i in range(len(cells)):
            flips[i], flops[i], grays[i] = Data.get_flip_flops(cells[i], Data.strobe_down, Data.stim_up, Data.Bframe_up)
            #delta_f_flips[i], delta_f_flops[i], delta_f_grays[i] = calc_stim_delta_f(flips[i], flops[i], grays[i], mode = 'mean')
        
        #plotData(flips[0].values(), 3, (0,6))
        #plotData(grays[0].values(), 3, (0,6))
    elif encoding == 'fast':
        timeCodes = GetCodeList(channels[2], [channels[2], channels[3]], threshold)
    
        flopTimeStamps = GetTimeStamps(1, timeCodes)
        flipTimeStamps = GetTimeStamps(2, timeCodes)
           
        signalChannel = channels[0]
        timingChannel = channels[1]
        codeChannels = [channels[2], channels[3]]
    
        SRP = SRPdecoder()
        timeCodes = SRP.GetCodeList(signalChannel, codeChannels)
        flopTimeStamps = SRP.GetTimeStamps(1, timeCodes)
        flipTimeStamps = SRP.GetTimeStamps(2, timeCodes)
        flipLengths, flopLengths = SRP.GetStimLengths(flipTimeStamps, flopTimeStamps)
        avgLength = SRP.AvgStimLength([flipTimeStamps, flopTimeStamps])
        
        SRP.stimLength = avgLength
        
        stimsPerSession = SRP.StimsPerSession(flipLengths, flopLengths, avgLength)

        for i in range(len(cells)):
            flips[i], flops[i], grays[i], sessions[i] = SRP.GetStimLists(cells[i], stimsPerSession, avgLength, flipTimeStamps, flopTimeStamps, Data.Bframe_up)
        
        #get means of truncated lsist
        session_avgs = list()
        plt.figure(20)
        for cell in sessions:
            temp_session_avg = np.nanmean(np.array(sessions[cell].values()), axis = 0)
            session_avgs.append(temp_session_avg)
            plt.plot(temp_session_avg+(cell*0.1))
            
        session_avgs_sorted = sorted(session_avgs, key=lambda x: x.argmax())
        plt.figure(21)
        for i in range(len(session_avgs_sorted)):        
            plt.plot(session_avgs_sorted[i]+(i*0.1))
            
#==============================================================================
#         flip_avgs = dict()
#         for cell in flips:
#             for block in flips[cell]:
#                 this_avg = np.nanmean([x[0:min([len(z) for z in flips[cell][block].values()])] for x in flips[cell][block].values()], axis = 0)
#                 flip_avgs[cell] = this_avg
#                 plt.plot(this_avg)
#             
#     
#     #chop flips and flops down to min length
#     for arr in flips:
#         min_len = min([len(i) for i in flips[arr].values()])
#         for ts in flips[arr]:
#             flips[arr][ts] = flips[arr][ts][0:min_len]
#     
#     for arr in flops:
#         min_len = min([len(i) for i in flops[arr].values()])
#         for ts in flops[arr]:
#             flops[arr][ts] = flops[arr][ts][0:min_len]
#     
#     for arr in grays:
#         min_len = min([len(i) for i in grays[arr].values()])
#         for ts in grays[arr]:
#             grays[arr][ts] = grays[arr][ts][0:min_len]
#     
#     flip_stats = dict()
#     flop_stats = dict()
#     gray_stats = dict()
#     flip_prestims = dict()
#     flop_prestims = dict()
#     
#     flip_thresh = dict()
#     flop_thresh = dict()
#     gray_thresh = dict()
#     
#     #frames to remove fron end in ANOVA
#     frames = 1
#         
#     #delete bad ROI
# #    del flips[168]
# #    del flops[168]
# #    del grays[168]
# 
#     
#     for i in range(len(flips)):
#         #get rid of bad first array
#         remove_block(flips[i], 0)
#         remove_block(flops[i], 0)
#         remove_block(grays[i], 0)
# #        remove_block(flips[i], 0)
# #        remove_block(flops[i], 0)
# #        remove_block(grays[i], 0)
# #        remove_block(flips[i], 3)
# #        remove_block(flops[i], 3)
# #        remove_block(grays[i], 3)
#         
# #        flip_prestims[i] = get_prestim_values(cells, flips, i)
# #        flop_prestims[i] = get_prestim_values(cells, flops, i)
#         
#         
#         #remove end frames
#         flipv = np.array(flips[i].values())[:-frames]
#         flopv = np.array(flops[i].values())[:-frames]
#         grayv = np.array(grays[i].values())[:-frames]
# #        flip_prev = np.array(flip_prestims[i].values())[:,frames:]
# #        flop_prev = np.array(flop_prestims[i].values())[:,frames:]
#         
# #        flip_vals = np.concatenate((flipv, flip_prev), axis = 0)
# #        flop_vals = np.concatenate((flopv, flop_prev), axis = 0)
#         
#         thr = 1.0
#         flip_truths = [f > thr for f in flipv]
#         flip_thresh[i] = [True in X for X in flip_truths]
#         flop_truths = [f > thr for f in flopv]
#         flop_thresh[i] = [True in X for X in flop_truths]
#         gray_truths = [f > thr for f in grayv]
#         gray_thresh[i] = [True in X for X in gray_truths]
#         
#         #test case
#         flip_vals = flipv
#         flop_vals = flopv
#         gray_vals = grayv
#            
#         #flop_vals = np.array(np.concatenate((flops[i].values(), flop_prestims[i].values()), axis = 0)).astype(np.float64)
#         #flip_vals = np.array(np.concatenate((flips[i].values(), flip_prestims[i].values()), axis = 0)).astype(np.float64)
#         
#         flip_vals = flip_vals.transpose()
#         flop_vals = flop_vals.transpose()
#         gray_vals = gray_vals.transpose()
#         
#         flip_stats[i] = stats.kruskal(*flip_vals)
#         flop_stats[i] = stats.kruskal(*flop_vals)
#         gray_stats[i] = stats.kruskal(*gray_vals)
#     
#     #Graphing Procedure
#     for i in range(len(cells)):
#         cells[i] = np.nan_to_num(cells[i])
#     
#     #FLIPS
#     r_flip_keys, nr_flip_keys = is_responsive(flip_stats, cells)
#     r_flips = [cells[i] for i in r_flip_keys]
#     nr_flips = [cells[i] for i in nr_flip_keys]
#     
#     if len(r_flips) == 0:
#         r_flip_keys = nr_flip_keys
#         r_flips = nr_flips
#         print "NO RESPONSIVE FLIPS"
#     
#     
#     r_flip_thresh_keys = [i for i in flip_thresh if True in flip_thresh[i]]
#     r_flop_thresh_keys = [i for i in flop_thresh if True in flop_thresh[i]]
#     r_gray_thresh_keys = [i for i in gray_thresh if True in gray_thresh[i]]
#    
#     
#     #FLOPS
#     r_flop_keys, nr_flop_keys = is_responsive(flop_stats, cells)
#     r_flops = [cells[i] for i in r_flop_keys]
#     nr_flops = [cells[i] for i in nr_flop_keys]
#     
#     if len(r_flops) == 0:
#         r_flop_keys = nr_flop_keys
#         r_flops = nr_flops
#         print "NO RESPONSIVE FLOPS"
#     
#     r_gray_keys, nr_gray_keys = is_responsive(gray_stats, cells)
#     r_grays = [cells[i] for i in r_gray_keys]
#     nr_grays =[cells[i] for i in nr_gray_keys]
#     
#     if len(r_grays) == 0:
#         r_gray_keys = nr_gray_keys
#         r_grays = nr_grays
#         print "NO RESPONSIVE GRAYS"
#     
#     #BOTH
#     r_both_keys = list(set(r_flip_keys) | set(r_flop_keys) | set(r_gray_keys))
#     nr_both_keys = list(set(nr_flip_keys) & set(nr_flop_keys) & set(nr_gray_keys))
#     r_both = [cells[i] for i in r_both_keys]
#     nr_both = [cells[i] for i in nr_both_keys]
#     
#     #FLIP AVGS
#     r_flip_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in r_flip_keys}
#     nr_flip_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in nr_flip_keys}
#     r_flop_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in r_flop_keys}
#     nr_flop_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in nr_flop_keys}
#     r_gray_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in r_gray_keys}
#     nr_gray_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in nr_gray_keys}
#     
# 
#     max_flip_val = max(max([np.max(i) for i in r_flip_avgs]), max([np.max(i) for i in nr_flip_avgs]))
#     max_flop_val = max(max([np.max(i) for i in r_flop_avgs]),max([np.max(i) for i in nr_flop_avgs]))
#     min_flip_val = min(min([np.min(i) for i in r_flip_avgs]),min([np.min(i) for i in nr_flip_avgs]))
#     min_flop_val = min(min([np.min(i) for i in r_flop_avgs]),min([np.min(i) for i in nr_flop_avgs]))
# 
#     #BOTH AVGS
#     r_flip_both_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in r_both_keys}
#     nr_flip_both_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in nr_both_keys}
#     r_flop_both_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in r_both_keys}
#     nr_flop_both_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in nr_both_keys}
#     r_gray_both_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in r_both_keys}
#     nr_gray_both_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in nr_both_keys}
#     
#     all_flip_avgs = {i:np.mean(flips[i].values(), axis = 0) for i in flips.keys()}
#     all_flop_avgs = {i:np.mean(flops[i].values(), axis = 0) for i in flops.keys()}
#     all_gray_avgs = {i:np.mean(grays[i].values(), axis = 0) for i in grays.keys()}
#     
#     
#     r_both_avgs = {i:np.concatenate((r_flip_both_avgs[i], r_flop_both_avgs[i], r_gray_both_avgs[i]), axis = 0) for i in r_both_keys}
#     nr_both_avgs = {i:np.concatenate((nr_flip_both_avgs[i], nr_flop_both_avgs[i], nr_gray_both_avgs[i]), axis = 0) for i in nr_both_keys}
#     all_avgs = {i:np.concatenate((all_flip_avgs[i], all_flop_avgs[i], all_gray_avgs[i]), axis = 0) for i in all_flip_avgs.keys()}
#     
# #    r_avgs_sorted = sorted(r_both_avgs, key = lambda x: np.argmax(x))
# #    r_avgs_sorted.reverse()
#     
#     r_gray_avg_sorted_keys = sorted(r_gray_avgs, key = lambda key: np.argmax(r_gray_avgs[key]))
#     nr_gray_avg_sorted_keys = sorted(nr_gray_avgs, key = lambda key: np.argmax(nr_gray_avgs[key]))
#     
#     
#     r_avgs_sorted_keys = sorted(r_both_avgs, key = lambda key: np.argmax(r_both_avgs[key]))
#     r_avgs_sorted = list()
#     for key in r_avgs_sorted_keys:
#         r_avgs_sorted.append(r_both_avgs[key])
#     get_colormaps(r_avgs_sorted,r_avgs_sorted_keys, 0, 0.6)
#     plt.title('Responsive Cell Averages (ANOVA)').set_family('helvetica')
#     plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
#     plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]) + len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
#     plt.axis('tight')
#     plt.savefig('Responsive_Cell_Averages_(ANOVA).eps', format = 'eps', dpi = 1200)
#     
# #    nr_avgs_sorted = sorted(nr_both_avgs, key = lambda x: np.argmax(x))
# #    nr_avgs_sorted.reverse()
#     nr_avgs_sorted_keys = sorted(nr_both_avgs, key = lambda key: np.argmax(nr_both_avgs[key]))
#     nr_avgs_sorted = list()
#     for key in nr_avgs_sorted_keys:
#         nr_avgs_sorted.append(nr_both_avgs[key])
#     get_colormaps(nr_avgs_sorted, nr_avgs_sorted_keys, 0, 0.6)
#     plt.title('Non-Responsive Flip / Flop Averages (ANOVA)')
#     plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
#     plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]) + len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
#     plt.axis('tight')
#     
# 
# #    all_avgs_sorted = sorted(all_avgs, key = lambda x: np.argmax(x))
# #    all_avgs_sorted.reverse()
#     all_avgs_sorted_keys = sorted(all_avgs, key = lambda key: np.argmax(all_avgs[key]))
#     all_avgs_sorted = list()
#     for key in all_avgs_sorted_keys:
#         all_avgs_sorted.append(all_avgs[key])
#     get_colormaps(all_avgs_sorted, all_avgs_sorted_keys, 0, 0.6)
#     plt.title('All Cell Averages')
#     plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
#     plt.axvline(x = len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]) + len(r_flip_both_avgs[r_flip_both_avgs.keys()[0]]), color = 'green', ls = 'dashed')
#     plt.axis('tight')
#     
#     
# #    r_flip_avg_norms = sorted(r_flip_avgs, key = lambda x: np.argmax(x))
# #    r_flip_avg_norms.reverse()
#     r_flip_avg_sorted_keys = sorted(r_flip_avgs, key = lambda key: np.argmax(r_flip_avgs[key]))
#     r_flip_avg_sorted = list()
#     for key in r_flip_avg_sorted_keys:
#         r_flip_avg_sorted.append(r_flip_avgs[key])
#     get_colormaps(r_flip_avg_sorted, r_flip_avg_sorted_keys, 0, 0.6)
#     plt.title('Responsive Flip Averages (ANOVA)')
#     plt.axis('tight')
#     
# 
# #    nr_flip_avg_norms = sorted(nr_flip_avgs, key = lambda x: np.argmax(x))
# #    nr_flip_avg_norms.reverse()
#     nr_flip_avg_sorted_keys = sorted(nr_flip_avgs, key = lambda key: np.argmax(nr_flip_avgs[key]))
#     nr_flip_avg_sorted = list()
#     for key in nr_flip_avg_sorted_keys:
#         nr_flip_avg_sorted.append(nr_flip_avgs[key])
#     get_colormaps(nr_flip_avg_sorted, nr_flip_avg_sorted_keys, 0, 0.6)
#     plt.title('Non-Responsive Flip Averages (ANOVA)')
#     plt.axis('tight')
#     
#     #FLOP AVGS
# 
# #    r_flop_avg_norms = sorted(r_flop_avgs, key = lambda x: np.argmax(x))
# #    r_flop_avg_norms.reverse()
#     r_flop_avg_sorted_keys = sorted(r_flop_avgs, key = lambda key: np.argmax(r_flop_avgs[key]))
#     r_flop_avg_sorted = list()
#     for key in r_flop_avg_sorted_keys:
#         r_flop_avg_sorted.append(r_flop_avgs[key])
#     get_colormaps(r_flop_avg_sorted, r_flop_avg_sorted_keys, 0, 0.6)
#     plt.title('Responsive Flop Averages (ANOVA)')
#     plt.axis('tight')
#     
# 
# #    nr_flop_avg_norms = sorted(nr_flop_avgs, key = lambda x: np.argmax(x))
# #    nr_flop_avg_norms.reverse()
#     nr_flop_avg_sorted_keys = sorted(nr_flop_avgs, key = lambda key: np.argmax(nr_flop_avgs[key]))
#     nr_flop_avg_sorted = list()
#     for key in nr_flop_avg_sorted_keys:
#         nr_flop_avg_sorted.append(nr_flop_avgs[key])
#     get_colormaps(nr_flop_avg_sorted, nr_flop_avg_sorted_keys, 0, 0.6)
#     plt.title('Non-Responsive Flop Averages (ANOVA)')
#     plt.axis('tight')
#     
#==============================================================================

    root.mainloop()
    