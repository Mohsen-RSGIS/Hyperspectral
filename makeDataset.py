# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:21:08 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)
"""

import numpy as np
import nargoutController as noc
#-------------------------#-------------------------#--------------------------


def makeDataset(Image_DataCube, *args):
    
    """
    Mask: This can be a classified mask.
    """
    
    #----------------------------------------------------------------------
        
    [R, C, B] = np.shape(Image_DataCube);
    #----------------------------------------------------------------------
    
    if len(args)==2:
        Mask = args[0].copy()
        if (type(args[1])!=list):
            Backgrounds = [args[1]].copy()
        else:
            Backgrounds = args[1].copy()
    elif len(args)==1:
        Mask = args[0].copy()
        Backgrounds = []
    else:
        Mask = np.ones([R,C])
        Backgrounds = []
    #----------------------------------------------------------------------
    
    Full__Image_Dataset = np.reshape(Image_DataCube.T, [B, R*C])
    #----------------------------------------------------------------------
    
    All_Labels = np.unique(Mask).astype(int);  Labels = All_Labels.tolist()
    #----------------------------------------------------------------------
    
    Rows_Columns__Mask=()
    Labels_No=[]
    Mask__Elements=[]
    #----------------------------------------------------------------------
    
    for bg in Backgrounds:
        if (np.prod(np.shape(np.where(np.array(Labels)==bg)))!=0):
            #print(bg)
            Labels.remove(bg)
            b = np.where(Mask==bg);  Mask[b] = -9999        # -9999: Background
            
    L = len(Labels);   #print(Labels)
    #----------------------------------------------------------------------
    
    Labels_No = list(np.zeros(L))
    for i in range(L):
        Labels_No[i] = len(np.where(Mask==Labels[i])[0])
    #print('Labels_No = ' + str(Labels_No))
    #----------------------------------------------------------------------
    
    Masked__Image_Dataset=np.empty([B,0],dtype=int)
    Masked__Image_Dataset__List = []
    #----------------------------------------------------------------------
    
    Mask_start=np.zeros([len(Labels)],dtype=int)
    Mask_end=np.zeros([len(Labels)],dtype=int)
    #----------------------------------------------------------------------

    for i in range(L):
        
        Rows_Columns__Mask = Rows_Columns__Mask + np.where(Mask.T==Labels[i])
        Mask__Columns = Rows_Columns__Mask[2*i][:];  Mask__Rows = Rows_Columns__Mask[2*i+1][:];
        #------------------------------------------------------------------
        
        Mask__Elements.append(R*Mask__Columns + Mask__Rows)  # Converts the row and column to matrix element (column base counting)
        #==================================================================

        #-------------------------- Mask' Starts & Ends ...
        if (i==0):
            Mask_start[i] = 0;  Mask_end[i] = Labels_No[i] + 0
        else:
            Mask_start[i] = 0 + Mask_end[i-1];  Mask_end[i] = Labels_No[i] + Mask_end[i-1]
        #--------------------------
        
        #==================================================================
        
        Masked__Image_Dataset = np.append(Masked__Image_Dataset, Full__Image_Dataset[:,Mask__Elements[i]], axis=1)
        #--------------------------
        Masked__Image_Dataset__List.append(Full__Image_Dataset[:,Mask__Elements[i]])
        #==================================================================
        
        
    Start_End = [Mask_start, Mask_end]      # The start and end positions in dataset columns
    
    return noc.nargoutController(Masked__Image_Dataset, Labels, Start_End)