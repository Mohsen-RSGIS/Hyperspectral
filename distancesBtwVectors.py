# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:03:23 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)
"""

#import time
import numpy as np
#--------------------------------------------------------------------------

def distancesBtwVectors(X, Y, Method='Euclidean'):     # X and Y are NumPy arrays.
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
#    M = max(X.max(), Y.max())
#    
#    M_digits__No = int(math.log10(abs(M)))+1
#    
#    a = 10 ** (M_digits__No);   print(a)
#    
#    X = X / a ;  Y = Y / a
    #--------------------------
    
    #-------------------------- Matrix based algorithm ...
    
#    Xrep = np.reshape(np.matlib.repmat(X,1,VnY), [VnX*VnY, Pn])
#    
#    Yrep = np.reshape(np.matlib.repmat(Y,VnX,1), [VnX*VnY, Pn])
#    
#    Distances = np.sqrt(np.reshape(np.sum((Xrep-Yrep)**2,1), [VnX,VnY]))
    #--------------------------
    
    Distances = np.zeros([VnX,VnY])
     
    for i in range(0,VnX):
         #print(i)
         
         Vi = X[i,:];
         
         #-------------------------- Mini-Matrix based algorithm ...
#         Distances[i,:] = np.sqrt(np.sum((Vi-Y)**2, 1))
         #--------------------------
         
         for j in range(0,VnY):
                              
             Vj = Y[j,:];
                          
             Distances[i,j] = np.sqrt(np.sum((Vi-Vj)**2))
# =============================================================================
    
        
    
    
    
    
# ============================================== Previous Code ...
#     Angles = np.zeros([VnX,VnY])
#     
#     for i in range(0,VnX):
#         
#         print(i)
#         
#         Vi = X[i,:].tolist();
#         
#         for j in range(0,VnY):
#             
#             if i!=j:
#                 
#                 Vj = Y[j,:].tolist();
#                 
#                 Vij = np.dot(Vi,Vj); #print("Vij: " + str(Vij))
#                 
#                 Vi_L = np.sqrt(np.dot(Vi, Vi)); #print("Vi_L: " + str(Vi_L))
#                 Vj_L = np.sqrt(np.dot(Vj, Vj)); #print("Vj_L: " + str(Vj_L))
#                 
#                 ViVj = Vi_L*Vj_L; #print("ViVj: " + str(ViVj))
#                 
#                 Cos_A = Vij / ViVj; #print("Cos(A): " + str(Cos_A))
#                 
#                 Angles[i,j] = np.arccos(Cos_A)*180/math.pi
# =============================================================================
    
    
    
#--------------------------------------------------------------------------
    
#    t1 = time.process_time()
#    print("\n\n>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<\n")
#    print("    Processing Time: " + str(round(t1-t0, 2)) + " (sec)\n")
#    print(">>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<\n\n")

#    #input("Press Enter to exit...")

# =============================================================================

    return Distances

