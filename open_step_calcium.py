# -*- coding: utf-8 -*-

"""

Created on Mon Feb 06 07:43:50 2017



@author: jesse

"""



import open_calcium_data_fast as ocdfast

import StepCodeFile as scf







import numpy as np

from matplotlib import pyplot as plt

import calcium_imaging_data_fast as cidfast

from copy import deepcopy

from collections import OrderedDict



#Gets Step code file

code_file = scf.StepCodeFile('10241_day1_drift_001_data.bin',8,4,2000)

stims = code_file.get_stim_angles([0,45,90,135,180,225,270,315])



recording_freq = 2000 # samples/sec

numberOfFrames = 3250

frame_rate= 7.6

Bframe_start = 23289

Bframe_length = (numberOfFrames/frame_rate) * recording_freq

timestamp = code_file.data[1][Bframe_start:]

step_data = code_file.data[2][Bframe_start:]

Bframes = np.arange(0, Bframe_length, recording_freq*(1.0/frame_rate))



#make Bframe





#get intensity file

csvfiletype = 3

cellsmean, areas, xycoords, Data = ocdfast.openData(csvfiletype)



# calculate DeltaF/F

# Run Konnerth lab deltaF/F and EWMA filter - USAGE: (cells = raw intensity data)                      

t_0 = 0.8 #0.2

t_1 = 0.75

t_2 = 3

samplingfreq = 8 #30

dff, dff_ewma, dff_offset = cidfast.run_deltaf_ewma(cellsmean, t_0, t_1, t_2, samplingfreq)

cells = deepcopy(dff_ewma)



handler = cidfast.FileHandler()



#makes dictionary of [cellnumber][orientation][block]

gray_offset= 5.0 #seconds

gray_offset *= 2000.0

response_data = OrderedDict()

grays = OrderedDict()

response_indices = OrderedDict()

grays_indices = OrderedDict()

for cell in range(len(cells)):

    response_data[cell] = OrderedDict()

    grays[cell] = OrderedDict()

    response_indices[cell] = OrderedDict()

    grays_indices[cell] = OrderedDict()

    for stim in stims:

        response_data[cell][stim] = list()

        grays[cell][stim] = list()

        response_indices[cell][stim] = list()

        grays_indices[cell][stim] = list()

        for ts in stims[stim]:

            begin = float(ts[0])

            end = float(ts[1])

            begin_frame_time = handler.get_nearest(begin, Bframes)

            begin_gray_time = handler.get_nearest(begin - gray_offset, Bframes)

            

            end_frame_time = handler.get_nearest(end, Bframes)

            end_gray_time = handler.get_nearest(begin, Bframes)

            

            begin_frame = list(Bframes).index(begin_frame_time)

            begin_gray = list(Bframes).index(begin_gray_time)

            

            end_frame = list(Bframes).index(end_frame_time)

            end_gray = list(Bframes).index(end_gray_time)

            

            chunk = cells[cell][int(begin_frame):int(end_frame)]

            gray_chunk = cells[cell][int(begin_gray):int(end_gray)]

            

            response_data[cell][stim].append(chunk)

            grays[cell][stim].append(gray_chunk)

            

            response_indices[cell][stim].append([int(begin_frame),int(end_frame)])

            grays_indices[cell][stim].append([int(begin_gray),int(end_gray)])



#example: plots all 5 block of degree 45 orientation for cell 0

#cell_0_45 = response_data[0][45]

#for block in cell_0_45:

#    plt.plot(block)



#gets averages

response_avgs = OrderedDict()

for cell in response_data:

    response_avgs[cell] = OrderedDict()

    for orientation in response_data[cell]:

        cell_ori = response_data[cell][orientation]

        A = np.array(cell_ori)

        B = np.mean(A, axis = 0)

        response_avgs[cell][orientation] = B



#example: plots all 45 degree responses

#for cell in response_avgs:

#    plt.plot(response_avgs[cell][45])



#example: plots all orientations for cell 10

#for ori in response_avgs[10]:

#    plt.plot(response_avgs[0][ori])



# Check DeltaF/F

# Plot all traces (whole)

plt.subplots()

rangeX = len(dff_ewma)

for i in range(rangeX):

    plt.plot(dff_ewma[i]+i)



    

tempstims = response_indices[0]

tempgrays = grays_indices[0]

stim_onsetframes = list()

grays_onsetframes = list()

for jj in tempstims:

    for i in range(len(tempstims[jj])):

        stim_onsetframes.append([tempstims[jj][i][0], tempstims[jj][i][1]])

for kk in tempgrays:

    for i in range(len(tempgrays[kk])):

        grays_onsetframes.append([tempgrays[kk][i][0], tempgrays[kk][i][1]])



# plot estimated stim onsets

for i in range(len(stim_onsetframes)):

    plt.axvline(stim_onsetframes[i][0], color = 'g')

    plt.axvline(stim_onsetframes[i][1], color = 'b')

    plt.axvline(grays_onsetframes[i][0], color = 'k')

    plt.axvline(grays_onsetframes[i][1], color = 'r')





# Just trying to get a picture of possible "selectivity" for each cell - idea is to have 5 trials plotted for each ori, then overlay bold, black avg trace



def multiPlot(data,avgdata,nrows, ncols, celllist, yaxlimit):

    fig, axes = plt.subplots(nrows, ncols, sharex='all', sharey='all')

    

    cols = ['0', '45', '90', '135', '180', '225', '270', '315']

    rows = ['{}'.format(row) for row in celllist]

    for ax, col in zip(axes[0], cols):

        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows):

        ax.set_ylabel(row, rotation=0, size='large')

       

    for cellnum in range(nrows):

              

        temp=data[celllist[cellnum]]

        tempavg=avgdata[celllist[cellnum]]

        templist = list()

        templistavg = list()

        

        for ori in temp:

            templist.append(temp[ori])

            templistavg.append(tempavg[ori])

        for j in range(ncols):

            #for k in range(len(templist[j])):

            #    axes[cellnum,j].plot(templist[j][k]) 

            axes[cellnum,j].plot(templistavg[j], linewidth=1.5, color='k')

            axes[cellnum,j].set_ylim([0, yaxlimit])



#multiPlot(response_data,response_avgs,5, 8,range(100,106),1.0)



def plotStack(imgdir, imgname, roizipname):

    import matplotlib.image as mpimg

    import readroi as roizip    

    img = mpimg.imread(imgdir+imgname)

    plt.subplots()

    imgplot = plt.imshow(img)

    imgplot = plt.imshow(img, cmap="gray")#, origin='lower')

    #read ROI zip file from Fiji/ImageJ (magic wand objects)

    #you will need readroi.py file (on GitHub)



    a=roizip.read_roi_zip(imgdir + roizipname)

    

    for j in range(len(a)):

        ylist = [a[j][1][i][0] for i in range(len(a[j][1]))]

        ylist.append(ylist[0])

        xlist = [a[j][1][i][1] for i in range(len(a[j][1]))]

        xlist.append(xlist[0])

        plt.plot(xlist, ylist, linestyle = '-', linewidth=0.5)

        plt.annotate(str(j+1), xy=(1, 1), xytext=(xlist[i], ylist[i]), color='limegreen', fontsize=8)

        

imgdir = '/Users/jcoleman/Documents/PYTHON/calcium imaging/psychopyStim/testdata/'

imgname = 'STD_New Stack.tif'

roizipname = 'ROIset_10241_day1_drift_001_moco.zip'

plotStack(imgdir, imgname, roizipname)        

        