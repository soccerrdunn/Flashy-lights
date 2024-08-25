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


def  bin_freqs(filename, nbins=256, nchannels=8):
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

    #get the indexes of the nodes that have been clustered
    node_idx = get_node_idx(clustered_set, nbins, nchannels)
    #use the node indexes to get the indexes of all of the leafs
