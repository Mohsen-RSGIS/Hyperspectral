# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:10:41 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)
"""

import math
import numpy as np

# =============================================================================
#np.seterr(all='raise')
# =============================================================================


class MLC ( object ):
    """
    MLC: Maximum Likelihood Classifier
    """
    
    def __init__ (self, Image_Shape):   
        
        if (len(Image_Shape)==2):
            [self.R, self.C] = Image_Shape
            
        elif (len(Image_Shape==3)):
            [self.R, self.C, self.B] = Image_Shape
    #==========================================================================
        
    def classify(self, Image_Dataset, Trainings_DS, Trainings_NS, Trainings_LS):
        """ - Image_Dataset: A Bx(R*C) dataset.
              - R: Number of Rows, C: Number of Columns, B: Number of Bands/Features
        """
        #----------------------------------------------------------------------
       
        L = len(Trainings_NS)
        
        Trn_start=np.zeros([len(Trainings_LS)],dtype=int)
        Trn_end=np.zeros([len(Trainings_LS)],dtype=int)
        
        Class_Prob = Trainings_NS / np.sum(Trainings_NS)
        #--------------------------
        Log_Prob = np.zeros([L, self.R*self.C])
        
        for i in range(L):     # i: Class index
            
            #-------------------------- Trainings' Starts & Ends ...
            if (i==0):
                Trn_start[i] = 0;  Trn_end[i] = Trainings_NS[i] + 0
            else:
                Trn_start[i] = 0 + Trn_end[i-1];  Trn_end[i] = Trainings_NS[i] + Trn_end[i-1]
            #------------------------------------------------------------------
            
            Cov_i = np.cov (Trainings_DS[:,Trn_start[i]:Trn_end[i]])
            #--------------------------
            
            inv_Cov_i = np.linalg.inv (Cov_i)
            
            #-------------------------- Regularization...
            m = Cov_i.min();  M = Cov_i.max()
            
            m_digits = int(math.log10(abs(m)))+1;  M_digits = int(math.log10(abs(M)))+1
            
            a = 10 ** (0.5*(M_digits - m_digits))
            #--------------------------
            
            det_cov__db_a = np.linalg.det (Cov_i / a)     # db_a: divided by a;
            """  det(aA) = (a^n)*det(A);  A: n x n matrix or the rank of A """
            #--------------------------
            
            mu = Trainings_DS[:,Trn_start[i]:Trn_end[i]].mean(axis=1)
            #--------------------------
            
            Rank__Cov_i = np.linalg.matrix_rank(Cov_i)  # If complete, it is equal to the number of bands (i.e., B).
            #print(Rank__Cov_i)
            #--------------------------
            CONST = -0.5*(Rank__Cov_i*np.log(a)+np.log(det_cov__db_a)) - Rank__Cov_i*0.5*np.log(2.*np.pi)
            #------------------------------------------------------------------
            
            d = (Image_Dataset.T - mu).T
            #--------------------------
            
            Log_Prob [i, :] =  -0.5*(d * np.dot(inv_Cov_i, d)).sum(0) + np.log(Class_Prob[i]) + CONST
        #======================================================================

        WinnerClasses_ID = np.argmax(Log_Prob, 0)
        Classification_Vec = np.zeros(self.R*self.C)
        #--------------------------

        for i in range(L):
            I = np.where(WinnerClasses_ID == i)
            Classification_Vec[I] = Trainings_LS[i]
        #--------------------------
        
        return Classification_Vec
    #==========================================================================
    
    def testClassification(self, Classification_Vec, Tests_Vec):
        
        d = Tests_Vec - Classification_Vec;  TrueClassified_No = self.R*self.C-np.count_nonzero(d)
        
        LabeledSamples_No = np.count_nonzero(Tests_Vec)
        
        OA = round((TrueClassified_No / LabeledSamples_No)*100, 2)    # OA: Overall Accuracy
        
        return OA
    #==========================================================================
    
    def generateClassifiedMap(self, Classification_Vec, OA = '???'):
        
        Classified_Map_2D = np.reshape(Classification_Vec, [self.R, self.C]).T
    
        return Classified_Map_2D

#==============================================================================

#******************************************************************************
#******************************************************************************
#******************************************************************************


if __name__ == "__main__":
    
    from scipy.io import loadmat
    from matplotlib import pyplot as plt
    #--------------------------
    
    #******************** Loading Data from an already saved mat-file:
    
# =============================================================================
    x = loadmat('Indian_Pine.mat')
    
    Classes_List = x['Classes']
    Ground_Truth = x['Trained_Data']
    Image_DataCube = x['Pixels']
# =============================================================================
    
    #----- OR:
       
# =============================================================================
    
#    try:
#        for i in range(len(Var_Names)): del vars()[Var_Names[i]]
#        del Var_Names
#    except:
#        pass
#    #--------------
#    
#    import load_NPZ_Mat_Data
#    
#    #**********************************************************************
#    
#    #--------------------------#-------------------------- Load a Mat-file
#    
#    #Var_Names = load_NPZ_Mat_Data.loadData()   # OR:
#    Var_Names = load_NPZ_Mat_Data.loadData('C:/Users/m.ghamary/Desktop/HSI/Indian_Pine.mat')   # OR: Var_Names = loadData('C:\\Users\\m.ghamary\\Desktop\\HSI\\Indian_Pine.mat')
#    #--------------
#    from load_NPZ_Mat_Data import *
#    #--------------
#    
#    Classes_List = Classes;  del Classes
#    Ground_Truth = Trained_Data; del Trained_Data
#    Image_DataCube = Pixels; del Pixels
#    #--------------------------#-------------------------- OR:Load an NPZ-file
#     
##    #Var_Names = load_NPZ_Mat_Data.loadData()   # OR:
##    Var_Names = load_NPZ_Mat_Data.loadData('C:/Users/m.ghamary/Desktop/HSI/Indian_Pine.npz')   # OR: Var_Names = loadData('C:\\Users\\m.ghamary\\Desktop\\HSI\\Indian_Pine.mat')
##    #--------------
##    from load_NPZ_Mat_Data import *
#    #--------------------------#--------------------------
#    
#    #**********************************************************************
    
# =============================================================================
    
    #--------------------------------------------------------------------------
    
    import time, MakeTT as TT
    t0 = time.process_time()
    #--------------------------------------------------------------------------
    
    Backgrounds = [0,1,2,3,4,5,6]
    
    #******************** Training and Test samples random selection and creating their datasets:
    
    Trn_Tst = TT.MakeTT(Image_DataCube, Ground_Truth, Backgrounds)  # Backgrounds: list(range(0,5)) --> [0,1,2,3,4]
    #--------------------------
    
    [Trainings_DS, Trainings_NS, Trainings_LS, Trainings_Vec, Trainings_2D, Tests_DS, Tests_NS, Tests_Vec, Tests_2D, Labels_No] = Trn_Tst.makeTT('n', 100 )
    #--------------------------
    
     #Trn_Tst.makeTT('n', 60 )   # 60 samples out of the labeled samples of each class;    # Example: 'n', [25,34,56,73,64,123,243,98,178,256,342,453]   # Number
     #Trn_Tst.makeTT('p', 20 )   # 20% of the labeled samples of each class;               # Example: 'p', [25,34,56,73,64,23,43,48,78,56,32,53]         # Percent
     #--------------------------
     
     # Definitions:
         # DS: Data Set
         # NS: Number Set
         # LS: Label Set
         # Vec: Vector
         # 2D: Two Dimensional
    #--------------------------------------------------------------------------
    
    #============================ Plotting ============================
    
    plt.figure(1)
    plt.subplot(121); plt.imshow(Trainings_2D); plt.set_cmap('gist_ncar'); plt.title('Training Samples')    
    plt.subplot(122); plt.imshow(Tests_2D); plt.set_cmap('gist_ncar'); plt.title('Test Samples')
    #plt.show()
    #==================================================================
    
    #--------------------------------------------------------------------------
    
    
    #******************** Trainings' Starts & Ends ...
    
    import StartEnd as se
    
    SE = se.se(Trainings_NS)
    #--------------------------------------------------------------------------
    
    
    #******************** Creating class dataset from full dataset and start-end of labels:
    
    import classDataset as CDS
    
    Class_Dataset = CDS.classDataset(Trainings_DS, SE)
    #--------------------------------------------------------------------------
    
    
    #******************** Creating full dataset from datacube:
    
    import makeDataset as DS
    
    Image_Dataset = DS.makeDataset(Image_DataCube)
    #[R, C, B] = np.shape(Image_DataCube);
    #Image_Dataset = np.reshape(Image_DataCube.T, [B, R*C])     # Full Dataset
    #--------------------------------------------------------------------------
    
    
    #******************** Feature Selection:
    
    #import MaximumAngleDiscrimination as MAD;    Features = MAD.mad(Class_Dataset)    # OR   Image_Dataset
    #--------------------------
    import MaximumTangentDiscrimination as MTD;    Features = MTD.mtd(Class_Dataset)    # OR   Image_Dataset
    #--------------------------
    
    Features = Features[: min(len(Features), min(Trainings_NS)-1)]
    #--------------------------------------------------------------------------
    
    
    #******************** Maximum Likelihood Classification:
    
    ML_Classifier = MLC(list(np.shape(Trainings_2D)))
    #--------------------------
    
    Classification_Vec = ML_Classifier.classify(Image_Dataset[Features,:], Trainings_DS[Features,:], Trainings_NS, Trainings_LS)
    #--------------------------
    
    #[R,C,B] = np.shape(Image_DataCube);
    #GroundTrurth_Vec = np.reshape(Ground_Truth.T, [1, R*C])
    """ 'GroundTrurth_Vec' can be used instead of 'Tests_Vec' (in the following function) for testing the classification result.
        Obviously, this data includes both Training and Test samples.
    """
    
    OA = ML_Classifier.testClassification(Classification_Vec, Tests_Vec)
    #--------------------------
    
    Classified_Map_2D = ML_Classifier.generateClassifiedMap(Classification_Vec, OA)
    #--------------------------------------------------------------------------
    
    
    #============================ Plotting ============================
    
    plt.figure(2); plt.imshow(Classified_Map_2D); plt.set_cmap('gist_ncar');
    plt.title('Classification Map (OA = ' + str(OA) + ' %)');
    
    #-------------------------- Show the plots
    plt.show()
    
    #==================================================================
    
    
    #**************************************************************************
    
    t1 = time.process_time()
    print("\n\n>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<\n")
    print("    Processing Time: " + str(round(t1-t0, 2)) + " (sec)\n")
    print(">>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<\n\n")

    del t0, t1
    
    #input("Press Enter to exit...")
    
    #**************************************************************************