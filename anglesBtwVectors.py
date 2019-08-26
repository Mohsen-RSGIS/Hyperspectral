# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:51:27 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)
"""

#import time
import sys, math, numpy as np
#--------------------------------------------------------------------------

def anglesBtwVectors(X, Y, Unit='rad'):     # X and Y are NumPy arrays.
    """
    X: A 2D dataset (VnX, Pn) or (BX x P)
    Y: A 2D dataset (VnY, Pn) or (BY x P)
    """
    
    #t0 = time.process_time()
    #--------------------------
    
    X=np.array(X, ndmin=2)
    Y=np.array(Y, ndmin=2)
    #--------------------------    
    [VnX, Pn] = np.shape(X)   # VnX: Number of the Point Vectors of X
    [VnY, Pn] = np.shape(Y)
    
    #-------------------------- Regularization...
    M = max(X.max(), Y.max())
    
    M_digits__No = int(math.log10(abs(M)))+1
    
    a = 10 ** (M_digits__No)
    
    X = X / a ;  Y = Y / a
    #--------------------------
    
    XYdot = np.dot(X,Y.T)
    
    XLen = np.array([np.sqrt(np.diag(np.dot(X,X.T)))]);  YLen = np.array([np.sqrt(np.diag(np.dot(Y,Y.T)))])
    
    XLen_x_YLen = XLen.T * YLen
    #--------------------------
    Cosine_Angles = np.array(XYdot / XLen_x_YLen)
    #--------------------------
    SameVectors_RC = np.where(abs(Cosine_Angles-1) <= sys.float_info.epsilon)     # sys.float_info.epsilon  =  2.220446049250313e-16
    #--------------------------
    
    Cosine_Angles[SameVectors_RC] = 1
    
    Angles = np.arccos(Cosine_Angles)
    
    #Angles[SameVectors_RC] = 0
    #--------------------------
    
    if Unit == 'deg':
        Angles = Angles * (180/np.pi)
        
    elif Unit == 'grad':
        Angles = Angles * (200/np.pi)
    #--------------------------
        
        
#--------------------------------------------------------------------------
    
#    t1 = time.process_time()
#    print("\n\n>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<\n")
#    print("    Processing Time: " + str(round(t1-t0, 2)) + " (sec)\n")
#    print(">>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<\n\n")

#    #input("Press Enter to exit...")

# =============================================================================

    return Angles