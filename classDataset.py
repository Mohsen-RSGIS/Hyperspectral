# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 10:34:47 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)
"""

import numpy as np
#-------------------------#-------------------------#--------------------------

def classDataset(Image_Dataset, Labels_Start_End):
    
    [B, P] = np.shape(Image_Dataset)
    
    L = len(Labels_Start_End[0])
    
    S = Labels_Start_End[0]
    E = Labels_Start_End[1]
    
    Class_Dataset = np.zeros([B, L])
    
    for i in range(L):
        
        Class_Dataset[:, i] = np.mean(Image_Dataset[:, S[i]:E[i]], 1)
        
    
    return Class_Dataset