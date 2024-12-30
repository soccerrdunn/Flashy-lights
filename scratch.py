#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  scratch.py
#  
#  Copyright 2024  <rick@raspberrypi>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import librosa as lb
import numpy as np
import pandas as pd
import configparser
from scipy.spatial.distance import squareform
import sklearn.cluster as clstr
import matplotlib.pyplot as plt

filename = '/home/rick/Downloads/j-bells-jazz.mp3'

#scratch
#filename is self explanatory, nbins tells you how many bins to break the spectrograph
#frequencies to, and nchannels tells you how many channels to ultimately have at the end
#open the file to read the music information, load into music with sample rate sr
music, sr = lb.load(filename)

#compute the spectrogram with x-axis being the bins (as mels ~ human perception)
#and the y axis is time, z axis is power (intensity)
SG=lb.feature.melspectrogram(y=music, sr=sr, n_mels=nbins)

#compute the correlation matrix of the frequency bins and their derivative because
#human perception of music is tied as much to the rise and fall of sound as it is
#the frequency and intensity
mydata=pd.DataFrame(SG) 
corr_matrix=mydata.transpose().corr()
mydata_diff=mydata.diff(axis=1)
corr_matrix_diff=mydata_diff.transpose().corr()
#compute the average of the two correlation matrices, consider weighting with future efforts
#getting rid of nans from diff and subtracting from one to convert the correlation matrix
#to a squareform distance matrix, then converting from squareform to standard
mean_corr_data = (corr_matrix+corr_matrix_diff.fillna(value=0))/2
#distances=squareform(1-mean_corr_data) Agglomerative clustering uses the square form

#now that we have the correlation of the various frequencies, we're going to use
#heirchical clustering using hte correlation as a distance parameter to cluster the
#bins.
clustered_set=clstr.AgglomerativeClustering(n_clusters=nchannels, metric='precomputed', linkage='single').fit(1-mean_corr_data)
dims = SG.shape
SG_Nu=np.zeros((nchannels,SG.shape[1]))
for i in range(0, nchannels-1):
    x=0
    for j in range (0, nbins-1):
        if clustered_set.labels_[j] == i:
            SG_Nu[i]=SG_Nu[i]+SG[j]
            x=x+1
    SG_Nu[i]=SG_Nu[i]/x
thresh=np.mean(SG_Nu,1)

sync_info = np.zeros((nchannels+1,SG_Nu.shape[1]))
#consider re-writing this using the np.where function
for i in range(1,SG_Nu.shape[1]-1):
    #calculate the threshold and convert to integer to make life easier
    #later
    y = (np.greater(SG_Nu[:,i],thresh)).astype(int)
    #if the result is the same, we aren't going to do anythign and skip
    #writing into the sync info matrix, the key will be the timestamps
    #of 0 except the first will be removed at the end and only times 
    #where somethign occurs will there be data available to act on
    if np.any(y!=(sync_info[1:,i-1])):
        sync_info[0,i] = lb.samples_to_time(i,sr=sr)
        sync_info[1:,i]=y
np.delete(sync_info , np.where(sync_info[0] == 0), axis=1)

        
    



#plot spectorgaph
fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True)
D=lb.amplitude_to_db(np.abs(lb.stft(music)), ref=np.max)
img = lb.display.specshow(D, y_axis='linear', x_axis='time',sr=sr, ax=ax[0])

D=lb.amplitude_to_db(SG_Nu, ref=np.max)

#get the indexes of the nodes that have been clustered
node_idx = get_node_idx(clustered_set, nbins, nchannels)

def main(args):
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
