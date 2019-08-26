# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:55:55 2019

@author: Mohsen Ghamary Asl (m.ghamary@gmail.com)
"""

import numpy as np
import nargoutController as noc

# =============================================================================
#np.seterr(all='raise')
# =============================================================================


class MakeTT ( object ):		# TT: Training and Test samples
    """
    This code selects random training and test samples and creates data-sets.
    """
    def __init__ ( self, Image_DataCube, Ground_Truth, Backgrounds=0 ):
        """ - Image_DataCube: An RxCxB multi/hyper-dimensional image data-cube.
              - R: Number of Rows, C: Number of Columns, B: Number of Bands/Features (e.g. wavebands)
            - Ground_Truth: An RxC ground truth data matrix.
            - Backgrounds: The labels in Ground_Truth map that are considered as background and not participated in the processes.
        """
        
        self.Ground_Truth = Ground_Truth
        
        [self.R, self.C, self.B] = np.shape(Image_DataCube)
        #----------------------------------------------------------------------
        self.Image_Dataset = np.reshape(Image_DataCube.T, [self.B, self.R*self.C])
        #----------------------------------------------------------------------
        
        All_Labels = np.unique(Ground_Truth);  self.Trn_Labels = list(All_Labels)
        #--------------------------
        
        if (type(Backgrounds)!=list):
            bg = [0];  bg[0] = Backgrounds;
            del Backgrounds;  Backgrounds = [0]; Backgrounds[0] = bg;  del bg
        #--------------------------
        for bg in Backgrounds:
            try: self.Trn_Labels.remove([bg])
            except: pass
            b = np.where(Ground_Truth==bg);  Ground_Truth[b] = 0
        self.L = len(self.Trn_Labels)
        #----------------------------------------------------------------------
        
        self.Labels_No = list(np.zeros(self.L))
        for i in range(self.L):
            self.Labels_No[i] = len(np.where(Ground_Truth==self.Trn_Labels[i])[0])
        #print('Labels_No = ' + str(self.Labels_No))
        #======================================================================
        
    def makeTTn ( self, Trainings_NumberSet = 60 ):
        """
        ***
        """
        if (len(np.shape(Trainings_NumberSet))==0 or np.shape(Trainings_NumberSet)[0]==1):
            Trainings_NumberSet = list(Trainings_NumberSet * np.ones(self.L).astype(int))
            
        return Trainings_NumberSet
        #======================================================================
    
    def makeTTp ( self, Trainings_PercentSet= 20 ):
        """
        ***
        """
        Trainings_NumberSet = list(np.zeros(self.L, dtype=int))
        if (len(np.shape(Trainings_PercentSet))==0 or np.shape(Trainings_PercentSet)[0]==1):
            Trainings_PercentSet = list(Trainings_PercentSet * np.ones(self.L).astype(int))
        
        for i in range(self.L):
            Trainings_NumberSet[i] = int(Trainings_PercentSet[i]/100 * self.Labels_No[i])
        
        return Trainings_NumberSet
        #======================================================================
    
    
    def makeTT (self, Mode = 'n', Trainings_QSet=60):
        """
            - Mode: The mode of training/test samples selection from the ground truth matrix.   - n: number, p: percent
        """
        
        if (Mode=='n'):
            Trainings_NumberSet = self.makeTTn(Trainings_QSet)
        elif (Mode=='p'):
            Trainings_NumberSet = self.makeTTp(Trainings_QSet)
        #----------------------------------------------------------------------
        
        Rows_Columns__Ground_Truth=()
        Labels_No=[]
        Ground_Truth__Elements=[]
        #----------------------------------------------------------------------
        
        #Trainings_NumberSet=[]  #np.zeros([1, len(Labels)]).astype(int)
        Trn_rsn=[]
        Trainings_Dataset__Elements=[]
        Trainings_Dataset=np.empty([self.B,0],dtype=int)
        Trainings_Dataset__List = []
        Trainings_2D=np.zeros([self.R,self.C])
        Trn_Rows=[]
        Trn_Columns=[]
        Trn_start=np.zeros([len(self.Trn_Labels)],dtype=int)
        Trn_end=np.zeros([len(self.Trn_Labels)],dtype=int)
        #----------------------------------------------------------------------
        
        Tests_NumberSet=[]
        Tst_rsn=[]
        Tests__Dataset_Elements=[]
        Tests_Dataset=np.empty([self.B,0],dtype=int)
        Tests_Dataset__List = []
        Tests_2D=np.zeros([self.R,self.C])
        Tst_Rows=[]
        Tst_Columns=[]
        #----------------------------------------------------------------------

        for i in range(self.L):
            
            Rows_Columns__Ground_Truth = Rows_Columns__Ground_Truth + np.where(self.Ground_Truth.T==self.Trn_Labels[i])
            Ground_Truth__Columns = Rows_Columns__Ground_Truth[2*i][:];  Ground_Truth__Rows = Rows_Columns__Ground_Truth[2*i+1][:];
            #------------------------------------------------------------------
            Labels_No.append(len(Ground_Truth__Rows))
            #------------------------------------------------------------------
            Ground_Truth__Elements.append(self.R*Ground_Truth__Columns + Ground_Truth__Rows)  # Converts the row and column to matrix element (column base counting)
            #------------------------------------------------------------------
            Random_Permutation = np.random.permutation(Labels_No[i])
            #==================================================================
                        
            #-------------------------- Trainings' Starts & Ends ...
            if (i==0):
                Trn_start[i] = 0;  Trn_end[i] = Trainings_NumberSet[i] + 0
            else:
                Trn_start[i] = 0 + Trn_end[i-1];  Trn_end[i] = Trainings_NumberSet[i] + Trn_end[i-1]
            #--------------------------
            
            #==================================================================
            
            Trn_rsn.append(np.sort(Random_Permutation[0:Trainings_NumberSet[i]]))          #np.random.randint(0,Labels_No[i], Trainings_NumberSet[i])    # rsn: random sample numners
            #------------------------------------------------------------------
            Trainings_Dataset__Elements.append(Ground_Truth__Elements[i][Trn_rsn[i]])
            #------------------------------------------------------------------
            Trn_Rows.append(Trainings_Dataset__Elements[i] % self.R)
            #--------------------------
            Trn_Columns.append((np.floor(Trainings_Dataset__Elements[i] / self.R)).astype(int))
            #--------------------------
            Trainings_2D[Trn_Rows[i], Trn_Columns[i]] = self.Trn_Labels[i]
            #------------------------------------------------------------------
            Trainings_Dataset = np.append(Trainings_Dataset, self.Image_Dataset[:,Trainings_Dataset__Elements[i]], axis=1)
            #--------------------------
            Trainings_Dataset__List.append(self.Image_Dataset[:,Trainings_Dataset__Elements[i]])
            #==================================================================
            
            Tests_NumberSet.append(Labels_No[i] - Trainings_NumberSet[i])
            #--------------------------
            Tst_rsn.append(np.sort(Random_Permutation[Trainings_NumberSet[i]:]))
            #------------------------------------------------------------------
            Tests__Dataset_Elements.append(Ground_Truth__Elements[i][Tst_rsn[i]])
            #------------------------------------------------------------------
            Tst_Rows.append(Tests__Dataset_Elements[i] % self.R)
            #--------------------------
            Tst_Columns.append((np.floor(Tests__Dataset_Elements[i] / self.R)).astype(int))
            #--------------------------
            Tests_2D[Tst_Rows[i], Tst_Columns[i]] = self.Trn_Labels[i]
            #------------------------------------------------------------------
            Tests_Dataset = np.append(Tests_Dataset, self.Image_Dataset[:,Tests__Dataset_Elements[i]], axis=1)
            #--------------------------
            Tests_Dataset__List.append(self.Image_Dataset[:,Tests__Dataset_Elements[i]])
            #==================================================================
        
        Trainings_LabelSet = self.Trn_Labels
        
        Trainings_Vector = np.reshape(Trainings_2D.T, [1, self.R*self.C])
        Tests_Vector = np.reshape(Tests_2D.T, [1, self.R*self.C])
        
        return noc.nargoutController(Trainings_Dataset, Trainings_NumberSet, Trainings_LabelSet, Trainings_Vector, Trainings_2D, \
               Tests_Dataset, Tests_NumberSet, Tests_Vector, Tests_2D, Labels_No)

#==============================================================================

#******************************************************************************
#******************************************************************************
#******************************************************************************


if __name__ == "__main__":
    
    from scipy.io import loadmat
    from matplotlib import pyplot as plt
    #--------------------------
    
    #************** Loading Data from an already saved Mat-file or NPZ-file:
    
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
    
    import time
    t0 = time.process_time()
    #--------------------------------------------------------------------------
    
    Backgrounds = [0,1,2,3,4,5,6]
    
    #******************** Training and Test samples random selection and creating their datasets:
    
    Trn_Tst = MakeTT(Image_DataCube, Ground_Truth, Backgrounds)  # Backgrounds: list(range(0,7)) --> [0,1,2,3,4,5,6]
    #--------------------------
    
    [Trainings_DS, Trainings_NS, Trainings_LS, Trainings_Vec, Trainings_2D, Tests_DS, Tests_NS, Tests_Vec, Tests_2D, Labels_No] = Trn_Tst.makeTT('n', [100] )       # 'n', [25,34,56,123,243,98,178,256,342,453]
    #--------------------------
    
     #Trn_Tst.makeTT('n', 60 )       # Example: 'n', [25,34,56,73,64,123,243,98,178,256,342,453]   # Number
     #Trn_Tst.makeTT('p', 20 )       # Example: 'p', [25,34,56,73,64,23,43,48,78,56,32,53]         # Percent
     #--------------------------
     
     # Definitions:
         # DS: Data Set
         # NS: Number Set
         # LS: Label Set
         # Vec: Vector
         # 2D: Two Dimensional
    #--------------------------------------------------------------------------
    
    #============================ Plotting ============================
    
    plt.figure()
    plt.subplot(121); plt.imshow(Trainings_2D); plt.set_cmap('gist_ncar'); plt.title('Training Samples')    
    plt.subplot(122); plt.imshow(Tests_2D); plt.set_cmap('gist_ncar'); plt.title('Test Samples')
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