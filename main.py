#----------------------------------------------------------------
#Author: Richard Dunn
#
#Insert credits here
#
#
#This project is designed to generate flashing lights to music.

import librosa as lb
import numpy as np
import pandas as pd
import configparser
from scipy.spatial.distance import squareform
import sklearn.cluster as clstr


#usage example: myfile= cp['DEFAULT']['songpath']+cp['DEFAULT']['songlist'].split('\n')[1]

def  bin_freqs(filename, nbins=256, nchannels=8):
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





def get_node_leaf_idx (cluster, num_leafs, num_nodes):
    #this function takes a linkage matrix cluster, the number of leafs at the end of the
    #cluster and the number of clusters you want to bin into and returns (not sure yet?)

    i = 0
    branches=[]
    offset=len(cluster)
    branches=np.append(branches,cluster[offset-1][1])
    branches=np.append(branches,cluster[offset-1][0])
    branches=branches.astype(int)
    while len(branches) < num_nodes:
        #lets go through each value that we're iterating down through the tree and identify
        #which nodes we want to keep
        for k in branches:
            #if this isn't a leaf node (i.e. a junction node, we need to split it, do nothing if
            #it's a leaf node since a leaf node won't split by definition
            #each itteration we will either delete one and add two, incrementing by a total of
            #one, or if a leaf node, simply copy over
            if k >= num_leafs:
                #find where in the array (by converting to list) the index of the value we
                #are looking for is
                idx = branches.tolist().index(k)
                #delete that node that we're diggning into
                branches=np.delete(branches, idx, 0)
                branches=np.insert(branches, idx, cluster[k-num_leafs][0])
                branches=np.insert(branches, idx, cluster[k-num_leafs][1])


                #now, return branches if you've reached the size you are looking for
                if len(branches)==num_nodes:
                    return(branches)

    return([])








        current_branch=branches
        branches=[]
        for k in current_branch:
            print(k)
            if k<num_leafs:
                branches=np.append(branches,k)
            else:
                branches=np.append(branches,cluster[k-num_leafs][0]) #yes, I know it's leaves
                branches=np.append(branches,cluster[k-num_leafs][1])
        branches=branches.astype(int)

    last_branch=branches[-2:]
    while len(branches)>num_nodes
        current_branch


