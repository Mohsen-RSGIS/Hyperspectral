# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:03:56 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)
"""

import numpy as np
import makeDataset as DS
from distancesBtwVectors import distancesBtwVectors
from matplotlib import pyplot as plt
#-------------------------#-------------------------#--------------------------

# K-Means Clustering...

def kmeans(Image_DataCube, K=2, I=4):
    """
        K: 
            - If int: Number of clusters. 
            - If 1D list or array: Base point numbers (as cluster centers) on the Image_Dataset columns.
            - If 2D list or array: Base points' list or array (as cluster centers) out of the Image_Dataset 
                                                    (maintaining its rows number (i.e. B: Number of the image bands) and varying on its columns).
        
        I: Number of iterations.
    """
    #-----------------------------------------------------------------
    
    [R,C,B] = np.shape(Image_DataCube)
    #-----------------------------------------------------------------
    
    Image_Dataset = DS.makeDataset(Image_DataCube)
    #-----------------------------------------------------------------
    
    if (np.ndim(K)==0):
        K = [K]
    #-----------------------------------------------------------------
    
    if  np.prod(np.shape(K))==1:      #  OR -->  (np.ndim(K) * np.shape(K))[0]==1
        K = K[0]
        BPs = np.random.permutation(R*C)[:K]  # Random base points (as cluster centers) selection
        Base_Points = Image_Dataset[:,BPs]
        print("\n\nBase point numbers: " + str(BPs) + "\n\n")
    #-------------------------
    elif np.prod(np.shape(K))>1:
        
        if np.shape(np.array(K, ndmin=2))[0]==1:
            K = np.array(K).tolist()
            BPs = K;  K = len(K)
            Base_Points = Image_Dataset[:,BPs]
        #-------------------------    
        elif np.shape(np.array(K, ndmin=2))[0]>1:
            if(type(K)==list):
                K = np.array(K)
            #-------------------------    
            Base_Points = K.copy();  K = np.shape(K)[1]
    #-----------------------------------------------------------------
    
    for i in range(I):
        
        print("Iteration: " + str(i+1))
        
        Distances = distancesBtwVectors(Base_Points.T, Image_Dataset.T)
        
        L = np.argmin(Distances, 0)
        
        if i<(I-1):
            
            for k in range(K):
                e = np.where(L==k);  Base_Points[:,k] = np.mean(Image_Dataset[:,e[0]], 1)
    #-----------------------------------------------------------------
    Clustered_Vector = L + 1    # +1: Because the position starts from 0 and Labels are better to be started from 1.
    Clustered_Map_2D = np.reshape(Clustered_Vector, [R, C]).T
    #-----------------------------------------------------------------
    
    plt.figure(); plt.imshow(Clustered_Map_2D); plt.set_cmap('gist_ncar');
    plt.title('Clustered Map ( ' + str(K) + ' Clusters by ' + str(i+1) + ' Iterations )')
    
    #-------------------------- Show the plot
    plt.show()
    #-----------------------------------------------------------------
    
    return Clustered_Map_2D