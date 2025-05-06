#----------------------------------------------------------------
#Author: Richard Dunn
#
#Insert credits here
#
#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
#This project is designed to generate flashing lights to music.

import librosa as lb
import numpy as np
import pandas as pd
import configparser
from scipy.spatial.distance import squareform
import sklearn.cluster as clstr
import matplotlib.pyplot as plt
import csv

example_filename = '/home/rick/Downloads/j-bells-jazz.mp3'

#usage example: myfile= cp['DEFAULT']['songpath']+cp['DEFAULT']['songlist'].split('\n')[1]

def get_node_idx (cluster, num_leafs, num_nodes=8):
    #this function takes a linkage matrix cluster, the number of leafs at the
    # end of the cluster and the number of clusters you want to bin into and
    # returns branch node IDs

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

def get_node_leafs(cluster, num_leafs, nodes):
    #iteratively go through and get all of the leaves associated with the nodes from
    #within the cluster

    #for every node
    leafs=[]
    for i in nodes:
        print('i=')
        print(i)
        if i < num_leafs: #if the node is a leaf node
            leaf_list=[i]
        else:
            leaf_list=[]
            leaf_list=np.append(leaf_list,cluster[i-num_leafs-1][1])
            leaf_list=np.append(leaf_list,cluster[i-num_leafs-1][0])
            leaf_list=leaf_list.astype(int)
            all_leaf = False
            while not(all_leaf): #this will keep going through k so long as there are nodes
                all_leaf = True  # set true, if all are leafs it will never reset and exit
                print('all_leaf')
                for k in leaf_list:
                    print('k=')
                    print(k)
                    # if this isn't a leaf node (i.e. a junction node, we need to split it,
                    # do nothing if
                    # it's a leaf node since a leaf node won't split by definition
                    # each itteration we will either delete one and add two, incrementing by
                    # a total of
                    # one, or if a leaf node, simply copy over
                    if k >= num_leafs:
                        all_leaf = False #set false so we know at least one node is in the list
                        # find where in the array (by converting to list) the index of the value we
                        # are looking for is
                        idx = leaf_list.tolist().index(k)
                        # delete that node that we're diggning into
                        leaf_list = np.delete(leaf_list, idx, 0)
                        # and replace with that node's branches
                        leaf_list = np.insert(leaf_list, idx, cluster[k - num_leafs - 1][0])
                        leaf_list = np.insert(leaf_list, idx, cluster[k - num_leafs - 1][1])
                    else:
                        print(leaf_list)
        #now we need to append to our list of arrays for each node i
        leafs[i]=leaf_list


def  simple_flash_thresh(filename, nbins=256, nchannels=8):
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

    #now re-bin everything into SG_Nu, which is the re-binned schema created from the 
    #bin labels calculated in the clustering algorithm, average them out
    SG_Nu=np.zeros((nchannels,SG.shape[1]))
    for i in range(0, nchannels-1):
        x=0
        for j in range (0, nbins-1):
            if clustered_set.labels_[j] == i:
                SG_Nu[i]=SG_Nu[i]+SG[j]
                x=x+1
        SG_Nu[i]=SG_Nu[i]/x
        
    #now create your threshold value, simple methos is to just calculate the average
    #so that half the time each channel is on and half the time its off (simple)
    thresh=np.mean(SG_Nu,1)

    #next find the threshold values, timestamps and determine which channels should be 
    #"on" = 1, or "off" = 0
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
    
    #remove the instances where nothing happens (the time stamp was left at zero above
    np.delete(sync_info , np.where(sync_info[0] == 0), axis=1)
    
    #now write to a csv
    np.savetxt((filename+'.csv'), SG_Nu, delimeter=',')


def test_timing(filename)
    #This takes the filename file and creates a flashing light display 
    #currently using 8 channels (future work will add additional flashing
    #areas
    shapes = list()
    fig,ax = plt.subplots()
    shapes.insert(0,Rectangle((0.2,0.6),width=0.6,height=0.05,edgecolor='blue',facecolor='lightblue'))
    shapes.insert(1,Polygon([(0.0,0.2),(0.2,0.2),(0.1,0.7)],edgecolor='green',facecolor='lightgreen'))
    shapes.insert(2,Rectangle((0.2,0.2),width=.05,height=0.2,edgecolor='red',facecolor='red'))
    shapes.insert(3,Rectangle((0.7,0.2),width=.05,height=0.2,edgecolor='red',facecolor='red'))
    shapes.insert(4,Polygon([(0.5,0.5),(0.7,0.5),(0.6,0.3)],edgecolor='green',facecolor='lightgreen'))
    shapes.insert(5,Rectangle((0.3,0.2),width=.15,height=0.15,edgecolor='red',facecolor='red'))
    shapes.insert(6,Circle((0.6,0.3), radius=0.05,edgecolor='blue',facecolor='lightblue'))
    shapes.insert(7,Circle((0.8,0.3), radius=0.05,edgecolor='blue',facecolor='lightblue'))

    for i in range(0, len(shapes)):
        ax.add_patch(shapes[i])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.title('test')
    plt.show(block=False)

>>> rect.set_visible(False)
>>> plt.show(block=False)
>>> rect.set_visible(True)
>>> plt.show(block=False)
