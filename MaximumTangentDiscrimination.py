# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:36:46 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)

Reference: https://ieeexplore.ieee.org/document/6587529
"""

# Maximum Tangent Discrimination ...

import numpy as np
#--------------------------
from anglesBtwVectors import anglesBtwVectors
#--------------------------#--------------------------#--------------------------

def mtd(Image_Dataset):
        
        Features = []
        #--------------------------
        
        TangentsofAngles_Btw_ImageBands = np.array(np.tan(anglesBtwVectors(Image_Dataset, Image_Dataset)), ndmin=2)
        
        Tangents_Sum = np.sum(TangentsofAngles_Btw_ImageBands, 0)
        #--------------------------
        
        Features.append(np.argmax(Tangents_Sum))
        
        T = TangentsofAngles_Btw_ImageBands[Features]
        #--------------------------
        
        StopPoint_index = np.argsort(T[0])[1]
        #print("\n\nStop point: " + str(StopPoint_index) + "\n")
        #--------------------------
        
        while Features[-1] != StopPoint_index:
            
            Features.append(np.argmax(np.prod(T, 0)))
            
            T = TangentsofAngles_Btw_ImageBands[Features]
                        
        #print("\n\nSelected features: " + str(Features[:-1]) + "\n\n")
        
        return Features[:-1]