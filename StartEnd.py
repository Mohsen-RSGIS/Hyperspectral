# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:21:35 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)
"""

import numpy as np
#-------------------------#-------------------------#--------------------------

def se(Numbers_Set):
    
    """
    This code calculates the Starts & Ends of each case in a set with the number set of "Numbers_Set".
    """
    
    L = len(Numbers_Set)
    
    Start=np.zeros(L, dtype=int)
    End=np.zeros(L, dtype=int)
    
    for i in range(L):
        if (i==0):
            Start[i] = 0;  End[i] = Numbers_Set[i] + 0
        else:
            Start[i] = 0 + End[i-1];  End[i] = Numbers_Set[i] + End[i-1]
    
    #-------------------------
    Start_End = [Start, End]
    #-------------------------
    
    return Start_End