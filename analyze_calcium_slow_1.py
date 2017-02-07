# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:30:58 2016

@author: jcoleman
"""

import Tkinter as tk
import calcium_imaging_data_slow as cidslow

root = tk.Tk()
threshold = 2.5
root.update()

encoding = 'slow'
filetype_csv = 3 # 1 = rows-columns EWMA csv file; 2 = rows-columns RAWintDen csv file; 3 = FIJI ROI csv file
filetype_data = 1    # 1 = group _001 cells; 2 = _002 cells (169 ROIs)    
timestamp_file = 1 # 1 = "8-bit" event codes (i.e. Matlab vis-stim) ; 2 = "voltage" event codes (i.e. PsychoPy vis-stim)

#========================================================

Data = cidslow.FileHandler()
Data45 = cidslow.FileHandler()
Data135 = cidslow.FileHandler()    

if filetype_csv == 1 or filetype_csv == 2:
    print (' *** open a rows-columns CSV intensity data file ***')
    int_smooth_red = Data.open_intensity_csv()
    cells = Data.get_cells_from_smoothed(int_smooth_red)
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
        
        print('*** Get timestamps for T5 45deg ***')    
        event_data45 = Data45.open_event_csv()
        channels45 = Data45.get_channels(event_data45)    
        Data45.get_event_stamps(channels45, threshold)
        
        print('*** Get timestamps for T5 135deg ***')
        event_data135 = Data135.open_event_csv()
        channels135 = Data135.get_channels(event_data135)    
        Data135.get_event_stamps(channels135, threshold)

# Concatenated stack data
# Calculate deltaF/F        
t_0 = 0.2
t_1 = 0.75
t_2 = 3
samplingfreq = 30

dff_45, dff_ewma_45, dff_offset_45 = cidslow.run_deltaf_ewma(cells45, t_0, t_1, t_2, samplingfreq)
dff_135, dff_ewma_135, dff_offset_135 = cidslow.run_deltaf_ewma(cells135, t_0, t_1, t_2, samplingfreq)     

dff_flips45, dff_flops45, dff_grays45 = cidslow.get_flip_flop_gray(dff_ewma_45, Data45)
dff_flips135, dff_flops135, dff_grays135 = cidslow.get_flip_flop_gray(dff_ewma_135, Data135)

dff45_flip_avgs, dff45_flop_avgs, dff45_gray_avgs, dff45_avgs, normdff45_all_avgs, dff45_avgs_sorted_keys, normdff45_avgs_sorted_keys, dff45_avgs_sorted, normdff45_avgs_sorted = cidslow.avgSort_FlipFlopGray(dff_flips45, dff_flops45, dff_grays45)
dff135_flip_avgs, dff135_flop_avgs, dff135_gray_avgs, dff135_avgs, normdff135_all_avgs, dff135_avgs_sorted_keys, normdff135_avgs_sorted_keys, dff135_avgs_sorted, normdff135_avgs_sorted = cidslow.avgSort_FlipFlopGray(dff_flips135, dff_flops135, dff_grays135)


# Plot all averaged and sorted data
cidslow.plotHeatAvgs(dff45_avgs_sorted, dff45_avgs_sorted_keys, dff_flips45, dff_flops45, .2,0.8, 'dF/F-EWMA | T5 001 45deg')
cidslow.plotHeatAvgs(dff135_avgs_sorted, dff135_avgs_sorted_keys, dff_flips135, dff_flops135, .2,0.8, 'dF/F-EWMA | T5 001 135deg')
# Plot data - Sort 135 data based on 45 data
temp135_avgs_sort45 = list()
for key in dff45_avgs_sorted_keys:
    temp135_avgs_sort45.append(dff135_avgs[key])
cidslow.plotHeatAvgs(temp135_avgs_sort45, dff45_avgs_sorted_keys, dff_flips135, dff_flops135, 0,1.0, 'dF/F-EWMA | T5 001 135deg (45deg sort)')
# Maybe concatenate 45-135 deg data for plotting?

# Plot normalized data
cidslow.plotHeatAvgs(normdff135_avgs_sorted, normdff135_avgs_sorted_keys, dff_flips135, dff_flops135,0.5,1,'norm. dF/F-EWMA | T5 001 135deg sorted')
cidslow.plotHeatAvgs(normdff45_avgs_sorted, normdff45_avgs_sorted_keys, dff_flips45, dff_flops45,0.5,1,'norm. dF/F-EWMA | T5 001 45deg sorted')

# Plot image and ROIs
import calcium_imaging_imageplots as cii

imgdir = '/Users/jcoleman/Documents/PYTHON/calcium imaging/sample data/'
imgfile = 'test.tif'
colormap = 'gray'
cii.makeTIFFfig(imgdir,imgfile,colormap)



#cidslow.figSlowStimTrace(dff135_avgs_sorted,.1)
#    
#max(map(max, dff45_avgs_sorted))
#max(map(max, dff135_avgs_sorted))
#min(map(min, dff45_avgs_sorted))
#min(map(min, dff135_avgs_sorted))

##Plot thresholded data
#dff45_avgs_sorted_TH, dff45_avgs_sorted_TH_keys = cidslow.dff_threshold_data(dff45_avgs_sorted, dff45_avgs_sorted_keys, 0.2)
#dff45_avgs_sorted_THnorm = deepcopy(dff45_avgs_sorted_TH)
#for key in range(len(dff45_avgs_sorted_THnorm)):
#    dff45_avgs_sorted_THnorm[key] -= min(dff45_avgs_sorted_THnorm[key])
#    dff45_avgs_sorted_THnorm[key] /= max(dff45_avgs_sorted_THnorm[key])
#    dff45_avgs_sorted_THnorm[key] = np.array(dff45_avgs_sorted_THnorm[key]).astype(np.float)
#cidslow.plotHeatAvgs(dff45_avgs_sorted_THnorm, dff45_avgs_sorted_TH_keys, dff_flips45, dff_flops45, 0.0,1.0, 'T5 001 45deg sorted TH = 0.2')

#dff135_avgs_sorted_TH, dff135_avgs_sorted_TH_keys = cidslow.dff_threshold_data(dff135_avgs_sorted, dff135_avgs_sorted_keys, 0.2)
#cidslow.plotHeatAvgs(dff135_avgs_sorted_TH, dff135_avgs_sorted_TH_keys, dff_flips135, dff_flops135, 0.2,1.0, 'T5 001 135deg sorted TH = 0.2')



## Calculate z-scores
#A = dff45_avgs_sorted[0]
#Az = stats.zscore(A)
#B = deepcopy(dff45_avgs_sorted)
#Bz = stats.zscore(B)
#
#max(map(max, Bz))