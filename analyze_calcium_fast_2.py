# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:53:42 2016

@author: Jesse; Jason
"""   

import numpy as np
import matplotlib.pyplot as plt
import Tkinter as tk
from copy import deepcopy
import calcium_imaging_data_fast as cidfast
import open_calcium_data_fast as ocdfast

root = tk.Tk()
root.update()

csvfiletype = 3 # filetype: 1 = EWMA dff CSV; 2 = raw intensity CSV; 3 = FIJI ROI CSV
timestamp_file = 1 # 1 = MATLAB; 2 = PsychoPy

# Open and import intensity data
cells, areas, xycoords, Data = ocdfast.openData(csvfiletype)

# Open and import timestamp data
cells_mean = list()
for i in range(len(cells)):
    cells_mean.append(cells[i]/areas[i])
    
# Split 45-135 data
cells45=list()
for cell in range(len(cells_mean)):
    cells45.append(cells_mean[cell][0:6500])
    
# Split 45-135 data
cells135=list()
for cell in range(len(cells_mean)):
    cells135.append(cells_mean[cell][6500:13000])

Data45 = deepcopy(Data)
Data135 = deepcopy(Data)

#channels, dff, dff_ewma, dff_offset = ocdfast.processData(cells_mean, timestamp_file, Data)
print('<> 45deg timestamp file <>')
channels45, dff45, dff_ewma45, dff_offset45 = ocdfast.processData(cells45, timestamp_file, Data45)
print('<> 135deg timestamp file <>')
channels135, dff135, dff_ewma135, dff_offset135 = ocdfast.processData(cells135, timestamp_file, Data135)

#%%
# Chunk up the data by sessions, etc
#flips, flops, grays, sessions, flip_i, flop_i, gray_i = ocdfast.decodeData(channels, csvfiletype, dff_ewma, Data)
flips45, flops45, grays45, sessions45, flip_i45, flop_i45, gray_i45 = ocdfast.decodeData(channels45, csvfiletype, dff_ewma45, Data45)
flips135, flops135, grays135, sessions135, flip_i135, flop_i135, gray_i135 = ocdfast.decodeData(channels135, csvfiletype, dff_ewma135, Data135)


#%%
# Chunk up stim and gray sessions
stimsession_cell45, graysession_cell45 = cidfast.chunkSessions(sessions45, grays45)
stimsession_cell135, graysession_cell135 = cidfast.chunkSessions(sessions135, grays135)

# Combine STIM and GRAY session DATA (all sessions)
sessions_stimgray45, sessions_graystim45, gray_session_onset_i45, stim_session_onset_i45 = cidfast.combineGrayStim(stimsession_cell45, graysession_cell45, flop_i45, gray_i45)
sessions_stimgray135, sessions_graystim135, gray_session_onset_i135, stim_session_onset_i135 = cidfast.combineGrayStim(stimsession_cell135, graysession_cell135, flop_i135, gray_i135)

# ignore sessions 1 and 5 due to movement artifacts ([0,4])
sessions_ignore = [0,4]
for i in range(len(sessions_graystim45)):
    for j in range(len(sessions_ignore)):
        sessions_graystim45[i][sessions_ignore[j]][:] = np.nan


# Find the min length of graystim sessions for averaging, calc. means and sort
# NAN means not working???s
minlength_graystimsession45, avgs_graystim45, sorted_graystimavgs45, norm_graystimavgs45, sorted_graystimavgs_keys45, sorted_graystimavgs_axis_labels45 = cidfast.meanGrayStimSort(sessions_graystim45)
minlength_graystimsession135, avgs_graystim135, sorted_graystimavgs135, norm_graystimavgs135, sorted_graystimavgs_keys135, sorted_graystimavgs_axis_labels135 = cidfast.meanGrayStimSort(sessions_graystim135)


#%%
#==============================================================================        
# DATA VISUALIZATION
sessions_graystim = deepcopy(sessions_graystim45)
avgs_graystim = deepcopy(avgs_graystim45)
stim_session_onset_i = deepcopy(stim_session_onset_i45)
sorted_graystimavgs = deepcopy(sorted_graystimavgs45)
sorted_graystimavgs_keys = deepcopy(sorted_graystimavgs_keys45)
norm_graystimavgs = deepcopy(norm_graystimavgs45)
dff_ewma = deepcopy(dff_ewma45)
gray_i = deepcopy(gray_i45)
flips = deepcopy(flips45)

#%%
sessions_graystim = deepcopy(sessions_graystim135)
avgs_graystim = deepcopy(avgs_graystim135)
stim_session_onset_i = deepcopy(stim_session_onset_i135)
sorted_graystimavgs = deepcopy(sorted_graystimavgs135)
sorted_graystimavgs_keys = deepcopy(sorted_graystimavgs_keys135)
norm_graystimavgs = deepcopy(norm_graystimavgs135)
dff_ewma = deepcopy(dff_ewma135)
gray_i = deepcopy(gray_i135)
flips = deepcopy(flips135)

#%%
cidfast.plotSessionTraces(10, 9, sessions_graystim, avgs_graystim, stim_session_onset_i, 'deltaf/f | 45_T3_001 | gray-stim', mode='share xy')
cidfast.plotHeatAvgs(sorted_graystimavgs, sorted_graystimavgs_keys, stim_session_onset_i, 0, 1, 'df/f-EWMA | 45_T3_001')

#cidfast.plotSessionTraces(10, 7, norm_graystimavgs, avgs_graystim, stim_session_onset_i, 'norm. deltaf/f | 45_T3_001 | gray-stim', mode='share xy')
cidfast.plotHeatAvgs(norm_graystimavgs, sorted_graystimavgs_keys, stim_session_onset_i, 0.4, 1, 'Norm. df/f-EWMA | 45_T3_001')

#%%
# Plot all avg traces (whole)
plt.subplots()
for i in range(len(sorted_graystimavgs)):
    plt.plot(sorted_graystimavgs[i]+i*0.3)
plt.axvline(x = stim_session_onset_i, color = 'k') 

# To get an overview of entire imaging session - sessions with 100% correlation across cells are suspect (Jesse - how to "quantify" movement from traces (Chen-Helmchen, etc); exclude sessions)
# Plot all traces (whole)
plt.subplots()
for i in range(len(dff_ewma)):
    plt.plot(dff_ewma[i]+i)
# Overlay gray onsets
for i in range(len(gray_i)):
    plt.axvline(x = gray_i[i][0], color = 'k')
# Overlay flip onsets
for i in range(len(flips[0])):
    tempflip = flips[0][i].keys()[0][0]
    plt.axvline(x = tempflip, color = 'r')    

#plt.subplots(figsize=cm2inch(1.6, 2.9))
#plt.savefig('All_Cell_Averages_(ANOVA).eps', format = 'eps', dpi = 1200) 



