"""

==================================================

                VICTRE PIPELINE

==================================================

Author: Miguel A. Lago

              miguel.lago@fda.hhs.gov

        Vanday Bundu
              vanday.bundu@fda.hhs.gov

                                 DISCLAIMER

This software and documentation (the "Software") were

developed at the Food and Drug Administration (FDA) by

employees of the Federal Government in the course of

their official duties. Pursuant to Title 17, Section

105 of the United States Code, this work is not subject

to copyright protection and is in the public domain.

Permission is hereby granted, free of charge, to any

person obtaining a copy of the Software, to deal in the

Software without restriction, including without

limitation the rights to use, copy, modify, merge,

publish, distribute, sublicense, or sell copies of the

Software or derivatives, and to permit persons to whom

the Software is furnished to do so. FDA assumes no

responsibility whatsoever for use by other parties of

the Software, its source code, documentation or compiled

executables, and makes no guarantees, expressed or

implied, about its quality, reliability, or any other

characteristic. Further, use of this code in no way

implies endorsement by the FDA or confers any advantage

in regulatory decisions. Although this software can be

redistributed and/or modified freely, we ask that any

derivative works bear some notice that they are derived

from it, and any modified versions bear some notice that

they have been modified.

More information: https://github.com/DIDSR/VICTRE_PIPELINE

"""
from asyncore import write
import numpy as np
import math
import copy
from scipy.ndimage import gaussian_filter
import random
from DetectBoundary import DetectBoundary
import Vess
from Bias import Bias
import writetoraw_forAndrea as write
from numpy.linalg import norm
import time
import cProfile
import pstats
from pstats import SortKey
#from MHD_Reader import MHD_Reader
import array as arr
import sys

class TumorSim:
    """
        Object contructor for the Victre Pipeline 
        :param anatomy_file: its a zipped raw file of the cell anatomy the tumor grows in
        :param mhd_file: An mhd file that house all the metadata explaining the anatomy_file
        :param loc: a .loc file that denotes the location of where tumor should grow
        :param size: An int that determines the size of the tumor and the environemnt 
        :param time_array: an array that houses the different time point that the tumor records(unit: iterations)
        :param seed: the seed value that determines the filename that will be outputed 
        :param locations: contains a list of location that the tumor will grow in (x,z,y)
    """
    def __init__(self, phantom, output, size, time_array, seed, locations) -> None:

        
        
         # ================== Create Vectors ===================
        # The domain of simulations is a cube sized 1 cm^3
        # I assume the size of each tumor cell is 50 micrometers
        # Hence, in each cube length are 200 grids (or max 200 cells)
        self.leng = 10
        self.N = size
        self.slen = self.N*self.N
        self.wlen = self.N*self.N*self.N

        # ================= Create Vectors ==================
        # Create vectors of intial values of all variables 
        self.nutr = np.ones((1,self.wlen))
        self.waste = np.ones((1,self.wlen))
        self.TAF = np.zeros((1,self.wlen))
        
        # ================ Initialization other varibales ======================
        
        self.pres = np.zeros(self.wlen, dtype=int)
        # Tumor cell activity should be zero at initial stage
        self.activity = np.zeros((1,self.wlen))
        
        self.celltype = np.zeros((1,self.wlen))[0]
        
        self.cell_energy = np.zeros((self.wlen))
        self.vess = [dict() for i in range(self.wlen)]
        self.vess_tag = np.array([[0]*self.wlen]*1).astype(float)
        self.vess_age = np.array([[0]*self.wlen]*1)

        self.hotpoint = np.array([[0]*self.wlen]*1)
        self.branchrecord = np.zeros((self.wlen))

        self.stackvalue = [0, 0]
        self.vessgrowth_flag = 0
        self.stackcount = 0

        X = [100, 135, 172, 183, 187, 181, 157, 123, 86, 52, 24, 19, 45, 18, 19]
        Y = [186, 179, 163, 123, 87, 47, 25, 15, 14, 23, 44, 148, 180, 77, 115]
        Z = np.array([101]*15)

        vindex = np.zeros(len(X), dtype=int)

        for i in range(0, len(X)):
            xx = round((X[i]/20)/0.05)
            yy = round((Y[i]/20)/0.05)
            zz = round((Z[i]/20)/0.05)

            vindex[i] = math.trunc(xx+yy*self.N+zz*self.slen)

        self.vess = [dict() for i in range(self.wlen)]
        for j in range(0,vindex.size):
            s = vindex[j]
            self.vess[s] = {'count':0, 'pare':[], 'son':[], 'direct':[]}

        for i in vindex:
            self.vess_tag[0][i] = 0.95
            self.vess_age[0][i] = 1

        self.vindexsave = vindex
        self.old_vindexsave = vindex

        self.xn = np.linspace(0,self.leng,self.N)

        [self.Y,self.X,self.Z] = np.meshgrid(self.xn,self.xn,self.xn)

        self.nod3xyz = np.array(np.vstack((self.Z.flatten(), self.Y.flatten(), self.X.flatten())).T) 

        self.nec = 1e-4

        self.counter1 = 0
        self.counter2 = 0

        self.spic = {
            "age": [[0]*self.wlen]*1,
            "dir": [[0]*self.wlen]*1,
            "par": [[0]*self.wlen]*1
                }
        self.spic_index = 0

        self.local_data = [[0]*self.wlen]*1

        self.sprout_index = []

        self.days_forAndrea = 0

        self.days = 0

        self.ana_no = 0
        
        self.mass_dim = 0

        self.totDays = 60 # Total simualation days (two months)
        self.tau = self.totDays*24*3600 # Unit: second
        self.L = 1e-2 # Unit: m, computational domain length: 10 mm
        self.k = 1/(self.leng*(self.N-1)) # Normalized time step = 0.0005. 2000 interation
        self.h = 1/(self.N-1) # Spatial step size for numerical method - central difference is used for convection term

        self.c0 = 4.3e-4 # Standard TAF concentration, unit: m^2/s [Wang & Li, 1998]           Original: 8e-14
        self.n0 = 8.4    # Standard nutrient concentration, unit: mol/m^3,  || Refer to "HEMOGLOBIN-BASED OXYGEN CARRIER HEMODILUTION AND BRAIN OXYGENATION"
        self.w0 = 10.5   # Nutrient consumption rate [Vaupel et al, 1987]

        # Waste Parameters
        self.Dn = 8e-14 # Nutrient diffusion rate, unit: m^2/s [Wang & Li, 1998]              %Origial : 8e-14
        self.rho_n0      = 6.8e-4  # Vessel nutrient supply rate, unit: mol/(m^3*sec) [Wang & Li, 1998] %Origial : 6.8e-4
        self.lambda_n0   = 3.0e-5  # Nutrient consumption rate [Vaupel et al, 1987]

        # Waste parameters
        self.Dw         = 4e-14  # Carbon dioxide diffusion coefficient, unit: m^2/s [estimated] %% Original: 4e-14
        self.rho_w0     = 1e-5  # Carbon dioxide secretion rate, unit: mol/(m^3s) [estimated]    %% Original: 1e-5
        self.lambda_w0  = 2.5e-5  # Carbon dioxide consumption rate, unit: ml/(cm^3s) [estimated]%% Original: 2.5e-5

        # TAF parameters
        self.Dc          = 1.2e-13  # TAF diffusion coefficient [estimated] 
        self.rho_c0      = 2e-9  # TAF secretion rate [estimated]
        self.lambda_c0   = 0  # VEGF natural decay rate, assumed to be very small [estimated]

        # Pressure
        self.pres_scale  = 1  # Tumor intersitial presssure: 1~60 mmHg, by adjusting this parameter, I can therefore investigate the growth patterns of different types of tumor, low pressure, high pressure.
                        # Our hypothesis is that the interstitial pressure inside solid tumor will give rise to different morphologies. Bigger interstitial pressure gives rise to dendritic tumor.
        self.cap_pres    = 30  # unit: mmHg
        self.p0          = 60*self.pres_scale  # unit: mmHg %% Change made by Aunnasha: Using 40 for round tumors as compared to 60 for dendritic tumors

        self.Kp          = 4.5e-15  # Hydraulic conductivity of the interstitium, unit: cm^2/mmHg-sec
        self.u0          = 5e-6
        self.pv          = self.cap_pres/self.p0  # Capillary/vascular pressure, unit: (20) mmHg 
        self.sigma0      = 0.15
        self.amplitude0  = 0.08*self.pres_scale  # Gaussian function amplitude

        self.k_AR2       = 500

        bias = Bias(self.N, self.slen)

        self.index_bias = bias.index_bias()

        self.pres_bias = bias.pres_bias()

        self.PHANTOM_MATERIALS = {

            "air": 0,

            "adipose": 1,

            "skin": 2,

            "glandular": 29,

            "nipple": 33,

            "muscle": 40,

            "paddle": 50,

            "antiscatter_grid": 65,

            "detector": 66,

            "ligament": 88,

            "TDLU": 95,

            "duct": 125,

            "artery": 150,

            "vein": 225

        }

        self.starttime = 100
        self.weight_old = 0
        self.act_old = 0

        self.cl = 0
        self.c2 = 0
        self.c3 = 0
        self.c4 = 0
        self.c5 = 0
        self.c6 = 0
        self.c7 = 0

        self.al = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 0
        self.a6 = 0
        self.a7 = 0

        self.sl = 0
        self.s2 = 0
        self.s3 = 0
        self.s4 = 0
        self.s5 = 0
        self.s6 = 0
        self.s7 = 0

        self.grad = False
        self.time_arr = copy.copy(time_array)
        self.time_array = sorted(time_array)

        self.seed = seed
        self.output = output
        

        self.full_anatomy = phantom
        
        self.coord = locations
        self.loc_it = 0

        




    def re_init(self):
        self.nutr = np.ones((1,self.wlen))
        self.waste = np.ones((1,self.wlen))
        self.TAF = np.zeros((1,self.wlen))
        
        self.pres = np.zeros(self.wlen, dtype=int)
        self.activity = np.zeros((1,self.wlen))
        
        self.celltype = np.zeros((1,self.wlen))[0]
        
        self.cell_energy = np.zeros((self.wlen))
        self.vess = [dict() for i in range(self.wlen)]
        self.vess_tag = np.array([[0]*self.wlen]*1).astype(float)
        self.vess_age = np.array([[0]*self.wlen]*1)

        self.hotpoint = np.array([[0]*self.wlen]*1)
        self.branchrecord = np.array([[0]*self.wlen]*1)

        self.stackvalue = [0, 0]
        self.vessgrowth_flag = 0
        self.stackcount = 0

        self.counter1 = 0
        self.counter2 = 0

        self.spic = {
            "age": [[0]*self.wlen]*1,
            "dir": [[0]*self.wlen]*1,
            "par": [[0]*self.wlen]*1
                }
        self.spic_index = 0

        self.local_data = [[0]*self.wlen]*1

        self.sprout_index = []

        self.days_forAndrea = 0

        self.days = 0

        self.ana_no = 0
        
        self.mass_dim = 0

        self.starttime = 199
        self.weight_old = 0
        self.act_old = 0

        self.cl = 0
        self.c2 = 0
        self.c3 = 0
        self.c4 = 0
        self.c5 = 0
        self.c6 = 0
        self.c7 = 0

        self.al = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 0
        self.a6 = 0
        self.a7 = 0

        self.sl = 0
        self.s2 = 0
        self.s3 = 0
        self.s4 = 0
        self.s5 = 0
        self.s6 = 0
        self.s7 = 0

        self.grad = False
        self.time_array = copy.copy(self.time_arr)
        self.time_array = sorted(self.time_array)

        

    
    def InitCancerCell(self):
        """
            Method that sets the starting postion of the tumor by 
            placing data points in the center
        """
        h = self.leng/(self.N-1)
        
        num_init_cell = 5
        init_size = 0.25
        
        # The constants here determine the intial location of the initial tumor cells
        index0 = 99*self.slen+99*self.N+100
        init_cell = self.nod3xyz[index0-1,0:]

        self.celltype[index0-1] = 1
        self.cell_energy[index0-1] = 15
        Mat_radius = [0.1793, 0.0928, 0.1199, 0.1402]
        Mat_thetal1 = [0.9848, 0.2331, 0.5748, 0.8741]
        Mat_thetal2 = [0.8167, 0.9745, 0.5811, 0.2300]

        if num_init_cell>1:
            for i in range(0, num_init_cell - 1):
                radius = Mat_radius[i]
                theta1 = Mat_thetal1[i]
                theta2 = Mat_thetal2[i]
                z = round(radius*math.sin(2*math.pi*theta1)/h)
                x = round(radius*math.cos(2*math.pi*theta1)*math.cos(2*math.pi*theta2)/h)
                y = round(radius*math.cos(2*math.pi*theta1)*math.sin(2*math.pi*theta2)/h)
                index = index0+x+y*self.N+z*self.slen
                init_cell = np.vstack((init_cell,self.nod3xyz[index-1,0:]))
                self.celltype[index-1] = 1
                self.cell_energy[index-1] = 15
                self.activity[0][index-1] = 1

        init_cell = init_cell*20

    def ReadAnatomyData(self, anatomy):
        """
            Method that reads the anatomy file of the tumor environment.
            In addition, it converts the data into a 3D array and sets 
            the multiplication factor for each respective element in the
            environment.

            :param anatomy: Anatomy array
            :returns: data array of the anatomy
        """
        fileID = open('Victre/Input_parameters.txt', 'r')
        A = fileID.readlines()
        self.days = int(A[0].rstrip())
        self.ana_no = int(A[1].rstrip())
        self.mass_dim = int(A[2].rstrip())


        ana = anatomy
        data = copy.deepcopy(ana)
        data = data.reshape(self.N*self.N*self.N)
        np.place(ana, ana != 88, 1)
        #Similiar case for this file as well 
        # Noted: Diffcult to gauge results with that of the one in MATLAB, determining 
        # a workaround
        ana_re = ana.reshape(self.N,self.N,self.N)

        ana_smooth = gaussian_filter(ana_re, 0.45)
        ligan = np.where(ana_smooth.flat>12)
        for elem in ligan:
            data[elem] = 88

        atmy_thickliga_filepath = 'Victre/results/anatomy' + str(self.ana_no) + 'thickliga.raw'
        fID = open(atmy_thickliga_filepath, 'w')
        data1 = np.array(data, dtype="uint8")
        data1.tofile(atmy_thickliga_filepath)
        

        datas = copy.deepcopy(data).astype(np.uint64)
        
        datas[np.where(data==self.PHANTOM_MATERIALS["adipose"])] = 10
        
        datas[np.where(data==self.PHANTOM_MATERIALS["glandular"])] = 30
        
        datas[np.where(data==self.PHANTOM_MATERIALS["ligament"])] = 1e6
        
        datas[np.where(data==self.PHANTOM_MATERIALS["duct"])] = 1
        
        datas[np.where(data==self.PHANTOM_MATERIALS["TDLU"])] = 1e6
        
        datas[np.where(data==self.PHANTOM_MATERIALS["artery"])] = 1e6
        
        datas[np.where(data==self.PHANTOM_MATERIALS["vein"])] = 1e6

        datas[np.where(data == self.PHANTOM_MATERIALS["skin"])] = 1e15

        datas[np.where(data == self.PHANTOM_MATERIALS["air"])] = 1e15
        
        datas[np.where((data != 1) & (data != 29) & (data != 88) & (data != 125) & (data != 95) & (data != 150) & (data != 225))] = 1

        datas = np.reshape(datas, (self.N, self.N, self.N))

        datas = np.reshape(datas,(self.N*self.N*self.N,))
        self.local_data = datas.transpose()
    
    def DetectBoundary(nod3xyz,leng):
        """
            Method that sets the boundary that the tumor can't
            grow past

            :param nod3xyz: the 3D matrix of the tumor environment
            :param leng: the leng that the boundary extends out within the matrix
            :returns: a dictionary that details the boundary of the tumor
        """
        geom = {
            "a": np.array(np.where(nod3xyz[0:,0] == 0)),
            "b": np.array(np.where(nod3xyz[0:,0] == leng)),
            "c": np.array(np.where(nod3xyz[0:,1] == 0)),
            "d": np.array(np.where(nod3xyz[0:,1] == leng)),
            "e": np.array(np.where(nod3xyz[0:,2] == 0)),
            "f": np.array(np.where(nod3xyz[0:,2] == leng))
                }

        geom['c'] = np.setdiff1d(geom['c'], geom['a'])
        geom['c'] = np.setdiff1d(geom['c'], geom['b'])
        
        geom['d'] = np.setdiff1d(geom['d'], geom['a'])
        geom['d'] = np.setdiff1d(geom['d'], geom['b'])
        
        geom['e'] = np.setdiff1d(geom['e'], geom['a'])
        geom['e'] = np.setdiff1d(geom['e'], geom['b'])
        geom['e'] = np.setdiff1d(geom['e'], geom['c'])
        geom['e'] = np.setdiff1d(geom['e'], geom['d'])
        
        geom['f'] = np.setdiff1d(geom['f'], geom['a'])
        geom['f'] = np.setdiff1d(geom['f'], geom['b'])
        geom['f'] = np.setdiff1d(geom['f'], geom['c'])
        geom['f'] = np.setdiff1d(geom['f'], geom['d'])

        return np.concatenate((geom['a'], geom['b'], geom['c'], geom['d'], geom['e'], geom['f']), axis=None)


    def Celldivide_anisotropic_anatomy_v3(self,s):
        """
            Method that calculates the pressure values whcih will
            determine tumor cell division behavior and doing so.

            :param s: the location of the cell where cell division is being determined
                      and potnetially occuring
                     
        """
        
        self.stackcount = 0
        self.nec = 1e-4
        prob = self.celltype[s+self.index_bias]
        index1 = np.where(prob > 0)
        index2 = np.where(prob == 0)

        # Generating an array that has 1 to indicate empty positions

        prob[index1] = 0
        prob[index2] = 1

        prob1 = self.local_data[s+self.index_bias]

        index1 = np.where(prob1 > 50)
        index2 = np.where(prob1 <= 50)

        # Generating an array that has 1 to indicate positions which have fat/grandular and 
        # 0 for other tissue types

        prob1[index1] = 0
        prob1[index2] = 1

        # Generating an array that has 1 to indicate locations that are empty and also
        # have fat/glandular tissues. 0 indicates the location is either filled or has stiffer tissues like ligaments.
        # This condition is applied mainly because there can be a
        # situation where all empty locations are either ligaments or glandular
        # tissues. In this case, we want the algorithm to move into the cellmove()
        # part of the algorithm, to create the fluid like spread along the ligaments
        # and glandular tissues.

        prob2 = prob * prob1

        if not np.any(np.where(prob2 == 1)):
            prob = np.ones(len(self.index_bias))
            scalar0 = (self.celltype[s + self.index_bias] != self.nec)
            prob = prob * scalar0

            if np.any(np.where(prob == 1)): # when are at least some locations which are non-necrotic
                pres_loc = self.pres[s+self.index_bias] # calculated pressure
                pres_dir = self.local_data[s+self.index_bias] # scalar multiplicative factors corresponding to each tissue type
                pres_var = pres_loc * pres_dir # manipulated pressure field 

                pres_var[np.where(pres_var == 0)] = 1

                pres_prob = 1/pres_var         # calculating proliferation probability as a simple inverse
                pres_prob = pres_prob * prob # If a cell is necrotic, the probability is 0
                pres_sum = np.sum(pres_prob)
                pres_prob = pres_prob/pres_sum # Normalizing the probability distribution 
                pres_csum = np.cumsum(pres_prob) # Calculating discrete CDF
                pres_csum = np.round(pres_csum,5)   # Approximating to the 5th decimal

                # Random Sampling

                pres_csum = np.insert(pres_csum, 0,0)
                mov = random.random()

                for i in range(0,len(pres_csum)):
                    if mov == 0:
                        gro=1
                        break
                    else:
                        if mov > pres_csum[i]:
                            if mov <= pres_csum[i + 1]:
                                gro = i
                                break
                # This statement is to ensure that in the off-chance that the
                # location with ligaments or stiffer materials get chosen, it
                # should be filled
                # if not 'gro' in locals():
                #    gro = 1

                if (self.celltype[s + self.index_bias[gro]] == 0): # if the randomly chosen location doesn't have a cell
                    self.celltype[s + self.index_bias[gro]] = 1
                    self.cell_energy[s] = 15
                else:
                    self.celltype[s + self.index_bias[gro]] = 1 # if the randomly chosen location already has a cell, then the cells need to move
                    self.cell_energy[s] = 15
                    old_cell_energy = self.cell_energy[s + self.index_bias[gro]]
                    self.cell_energy[s +  self.index_bias[gro]] = 15
                    self.stackvalue = [ s + self.index_bias[gro], old_cell_energy]
                    self.cellmove_v3()
        else:   # there are several empty locations, which are not ligaments or stiffer  tissues, to add cells
            pres_loc = self.pres[s + self.index_bias.T] # calculated pressure
            pres_dir = self.local_data[s + self.index_bias] # scalar multiplicative factors corresponding to each tissue type
            pres_var = pres_loc * pres_dir      # manipulated pressure fields

            pres_var[np.where(pres_var == 0)] = 1

            pres_prob = 1/pres_var # Calculating proliferation probability as a simple inverse
            pres_prob = pres_prob * prob2 # If a location is occupied and has stiffer tissue, probabaility is 0
            pres_sum = np.sum(pres_prob)
            pres_prob = pres_prob/pres_sum # Normalize the probabaility distribution 
            pres_csum = np.cumsum(pres_prob) # Calculating discrete CDF
            pres_csum = np.round(pres_csum,5) # Approximating to the 5th decimal 

            # Random Sampling
            pres_csum = np.insert(pres_csum, 0,0)
            mov = random.random()

            for i in range(0, len(pres_csum)):
                if mov==0:
                    gro=1
                    break
                else:
                    if mov>pres_csum[i]: 
                        if mov<=pres_csum[i+1]:
                            gro=i
                            break
            self.celltype[s + self.index_bias[gro]] = 1
            self.cell_energy[s] = 15
            self.cell_energy[s + self.index_bias[gro]] = 15

    def cellmove_v3(self): # Cells are pushing to move outward
        """
            Method used the push cell division outward. More specifically
            used to determine another location to cell divide when the 
            previous one was already occupied
        """
        # This function is used for tumor cell movement when inner cell proliferation
        self.stackcount = self.stackcount+1

        nec = 1e-4

        if self.stackcount <= 900:

            s = self.stackvalue[0]
            old_cell_energy = self.stackvalue[1]

            prob = self.celltype[s+self.index_bias]
            index1 = np.where(prob > 0)
            index2 = np.where(prob == 0)

            prob[np.array(index1)] = 0
            prob[np.array(index2)] = 1

            prob1 = self.local_data[s + self.index_bias]
            index1 = np.where(prob1 > 50)
            index2 = np.where(prob1 <= 50)

            prob1[index1] = 0
            prob1[index2] = 1

            prob2 = prob * prob1

            if not np.any(np.where(prob2 == 1)):
                prob = np.ones(len(self.index_bias))
                scalar0 = (self.celltype[s + self.index_bias] != self.nec)
                prob = prob * scalar0
            
            if np.any(np.where(prob == 1)):

                pres_loc = self.pres[s + self.index_bias]
                pres_dir = self.local_data[s + self.index_bias]
                pres_var = pres_loc * pres_dir
                pres_var[np.where(pres_var == 0)] = 1
                pres_prob = 1/pres_var
                
                pres_prob = pres_prob * prob


                pres_sum = np.sum(pres_prob)
                pres_prob = pres_prob/pres_sum
                pres_csum = np.cumsum(pres_prob)
                pres_csum = np.round(pres_csum,5)



                pres_csum = np.insert(pres_csum, 0,0)
                mov = random.random()


                for i in range(0, len(pres_csum)):
                    if mov == 0:
                        gro = 1
                        break
                    else:
                        if mov > pres_csum[i]:
                            if mov <= pres_csum[i + 1]:
                                gro = i
                                break
                
                if self.celltype[s + self.index_bias[gro]] == 0:
                    self.celltype[s + self.index_bias[gro]] = 1
                    self.cell_energy[s + self.index_bias[gro]] = old_cell_energy
                else:
                    self.celltype[s + self.index_bias[gro]] = 1
                    old_cell_energy_move = self.cell_energy[s + self.index_bias[gro]]
                    self.cell_energy[s + self.index_bias[gro]] = old_cell_energy
                    self.stackvalue = [s + self.index_bias[gro], old_cell_energy_move]
                    self.cellmove_v3()
    '''
    def cellmove_v4(self,s):

        self.stackcount = 0
        self.nec = 1e-4
        cont = True
        old_cell_energy = 15

        while self.stackcount <= 900:
            prob = self.celltype[s+self.index_bias]
            index1 = np.where(prob > 0)
            index2 = np.where(prob == 0)

            prob[np.array(index1)] = 0
            prob[np.array(index2)] = 1

            prob1 = self.local_data[s + self.index_bias]
            index1 = np.where(prob1 > 50)
            index2 = np.where(prob1 <= 50)

            prob1[index1] = 0
            prob1[index2] = 1

            prob2 = prob * prob1

            if not np.any(np.where(prob2 == 1)):
                prob = np.ones(len(self.index_bias))
                scalar0 = (self.celltype[s + self.index_bias] != self.nec)
                prob = prob * scalar0
                cont = False
                if not np.any(np.where(prob == 1)):
                    break
            
            if not np.any(np.where(prob == 1)):
                    break
            
            pres_loc = self.pres[s + self.index_bias]
            pres_dir = self.local_data[s + self.index_bias]
            pres_var = pres_loc * pres_dir
            pres_var[np.where(pres_var == 0)] = 1
            pres_prob = 1/pres_var
            
            if cont == True:
                pres_prob = pres_prob * prob2
            else:
                pres_prob = pres_prob * prob


            pres_sum = np.sum(pres_prob)
            pres_prob = pres_prob/pres_sum
            pres_csum = np.cumsum(pres_prob)
            pres_csum = np.round(pres_csum,5)

            pres_csum = np.insert(pres_csum, 0,0)
            mov = random.random()


            for i in range(0, len(pres_csum)):
                if mov == 0:
                    gro = 1
                    break
                else:
                    if mov > pres_csum[i]:
                        if mov <= pres_csum[i + 1]:
                            gro = i
                            break

            if cont == True and self.stackcount == 0:
                self.celltype[s + self.index_bias[gro]] = 1
                self.cell_energy[s] = 15
                self.cell_energy[s + self.index_bias[gro]] = 15
                
            if (self.celltype[s + self.index_bias[gro]] == 0): # if the randomly chosen location doesn't have a cell
                self.celltype[s + self.index_bias[gro]] = 1
                if self.stackcount == 0:
                    self.cell_energy[s] = old_cell_energy
                else:
                    self.cell_energy[s + self.index_bias[gro]] = old_cell_energy
                break
            else:
                self.celltype[s + self.index_bias[gro]] = 1 # if the randomly chosen location already has a cell, then the cells need to move
                if self.stackcount == 0:
                    self.cell_energy[s] = old_cell_energy
                
                #self.cell_energy[s +  self.index_bias[gro]] = old_cell_energy
                old_cell_energy = self.cell_energy[s + self.index_bias[gro]]
                s = s + self.index_bias[gro]
                #self.stackvalue = [ s + self.index_bias[gro], old_cell_energy]
                self.stackcount += 1
                #self.cellmove_v3()
    '''

    def CTP(self,cind):
        """
            Method to calculate cell-induced pressure

            :param cind: data points representing tumor cells
        """
        for tt in range(0, cind.size):
            s=cind[0,tt]

            # Density as defined by Equations (3c)
            density = np.sum(self.celltype[s+self.index_bias]>0)/len(self.index_bias)

            try:
                # Euclidean distance in Equation (2), (4a)
                # dist = sum((variables.nod3xyz[pres_bias,0:] - np.tile(variables.nod3xyz[s, 0:],(len(pres_bias),1)))**2) ** .5
                dist = np.array((np.sum((self.nod3xyz[s + self.pres_bias, 0:] - np.tile(self.nod3xyz[s, 0:],(len(self.pres_bias),1))) ** 2, axis=1))**.5)
            except IndexError:
                continue
            
            # Alpha in Equation (4b) is assumed to be constant
            amplitude=self.amplitude0

            # Sigma as defined by Equation (4c)
            sigma = (self.sigma0*density**2)/((density**2)+(0.5**2))+0.05

            test = (-dist**2)/(2*sigma**2)

            # Pressure based on Gaussian function as in Equation (4a)
            self.pres[s+self.pres_bias.T] += amplitude * np.exp(test)

    def VTP(self,vind):
        """
            Calculates vessel-induced pressure
            :param vind: data points representing vessel cells
        """
        for ss in range(0,vind.size):
            v = vind[0,ss]
            # Tumor cell density around vessel cell, here I assume vessel
            # only causes additional pressure inside tumor due to membrane
            # effect [Ref]

            density = np.sum(self.celltype[v+self.index_bias]>0)/len(self.index_bias)

            # Equation (4b), alpha0 set to be 0.01
            amplitude = (0.01*density**2)/(density**2 + 0.5**2)

            # Euclidean distance in Equation (4a)
            dist = np.array((np.sum((self.nod3xyz[v + self.pres_bias, 0:] - np.tile(self.nod3xyz[v, 0:],(len(self.pres_bias),1))) ** 2, axis=1))**.5)

             # Equation (4c)
            sigma = np.array((self.sigma0*density**2)/((density**2)+(0.5**2))+0.05)

             # Add vessel cell induced pressure
            self.pres[v+self.pres_bias.T] = self.pres[v+self.pres_bias.T] + amplitude * math.e ** ((-dist**2)/(2*sigma**2))

    def Gradient(self, t1):
        """
            Used to calculate Pressure Gradients
            :param t1: the tumor space that lies within the boundary
            :returns: x, y, and z coordinates of the pressure gradient
        """
        gradp = np.zeros((3,self.wlen))


        # Using Central difference Approximation
        # Note: These three variables have different values to MatLAB due to roundoff issues 
        gradp[0,t1 - 1] = (self.pres[t1 + 1] - self.pres[t1 - 1])/(2*self.h)
        gradp[1, t1 - 1] = (self.pres[t1+self.N] - self.pres[t1-self.N])/(2*self.h) # Number of points between Y1 and Y2 is N
        gradp[2,t1 - 1] = (self.pres[t1+ self.slen] - self.pres[t1-self.slen])/(2*self.h) # number of points saved between Y1 and Y2 is slen

        ux=-self.Kp*self.p0/(self.u0*self.L)*np.reshape(gradp[0,0:],(self.N,self.N,self.N))
        uy=-self.Kp*self.p0/(self.u0*self.L)*np.reshape(gradp[1,0:],(self.N,self.N,self.N))
        uz=-self.Kp*self.p0/(self.u0*self.L)*np.reshape(gradp[2,0:],(self.N,self.N,self.N))

        ux=np.reshape(ux,(self.N,self.N,self.N))
        uy=np.reshape(uy,(self.N,self.N,self.N))
        uz=np.reshape(uz,(self.N,self.N,self.N))

        return ux,uy,uz

    def Nutrients(self,weight, v_rad, boundary, ux, uy, uz):
        """
            Calculates nutrient concentration(mainly 02) within the tumor 
            environment
            :param weight: 
            :param v_rad: 
            :param boundary: boundary of the tumor environment 
            :param ux: the x component of the pressure gradient 
            :param uy: the y component of the pressure gradient
            :param uz: the z component of the pressure gradient
        """
        act = self.activity.reshape(self.N, self.N, self.N)
        v = self.vess_tag.reshape(self.N, self.N, self.N)
        c = self.celltype.reshape(self.N,self.N, self.N)
        n = self.nutr.reshape(self.N, self.N, self.N)


        #### Nutrients Equation ####
        if self.grad == True:
            self.c1 = 1-6*self.Dn*self.tau*self.k/((self.L**2)*(self.h**2))
            self.c2 = self.Dn*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*ux[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h)
            self.c3 = self.Dn*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*uy[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.c4 = self.Dn*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*uz[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.c5 = self.Dn*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*ux[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.c6 = self.Dn*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*uy[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.c7 = self.Dn*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*uz[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h)
            

        rho_n = self.rho_n0*v_rad*weight 
        lambda_n = self.lambda_n0*act[1:self.N-1,1:self.N-1,1:self.N-1]

        rho_nf = self.k*self.tau/self.n0*rho_n[1:self.N-1,1:self.N-1,1:self.N-1]* v[1:self.N-1,1:self.N-1,1:self.N-1]
        lambda_nf = self.k*self.tau/self.n0*lambda_n*c[1:self.N-1,1:self.N-1,1:self.N-1]
        c_n = copy.deepcopy(n)
        n[1:self.N-1,1:self.N-1,1:self.N-1] = ( self.c1*c_n[1:self.N-1,1:self.N-1,1:self.N-1] 
        + self.c2*c_n[2:self.N,1:self.N-1,1:self.N-1] 
        + self.c3*c_n[1:self.N-1,2:self.N,1:self.N-1] 
        + self.c4*c_n[1:self.N-1,1:self.N-1,2:self.N] 
        + self.c5*c_n[0:self.N-2,1:self.N-1,1:self.N-1] 
        + self.c6*c_n[1:self.N-1,0:self.N-2,1:self.N-1]
        + self.c7*c_n[1:self.N-1,1:self.N-1,0:self.N-2]
        + rho_nf - lambda_nf )

        temp_n=np.array(n.flat)
        temp_n[boundary]=1
        # temp_n = np.reshape(temp_n,(variables.N, variables.N, variables.N))
        self.nutr=copy.deepcopy(temp_n[0:].T)


    def Metabolic(self, weight, v_rad, boundary, ux, uy, uz):
        """
            Calculates metabolic waste concentration(mainly C02) within the tumor 
            environment
            :param weight: 
            :param v_rad:
            :param boundary: boundary of the tumor environment 
            :param ux: the x component of the pressure gradient 
            :param uy: the y component of the pressure gradient
            :param uz: the z component of the pressure gradient
        """
        n = self.nutr.reshape(self.N, self.N, self.N)
        w = self.waste.reshape(self.N, self.N, self.N)
        v = self.vess_tag.reshape(self.N, self.N, self.N)
        c = self.celltype.reshape(self.N, self.N, self.N)

        #### Metabolic Waste ####
        if self.grad == True:
            self.a1 = 1-6*self.Dw*self.tau*self.k/((self.L**2)*(self.h**2))
            self.a2=self.Dw*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*ux[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h)
            self.a3=self.Dw*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*uy[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.a4=self.Dw*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*uz[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.a5=self.Dw*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*ux[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.a6=self.Dw*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*uy[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.a7=self.Dw*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*uz[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h)

        w_new=w
        w_new[np.where(w_new < 1)]=1
        act= n/(n+1)*2*math.e**(-(w_new-1)**(4/.2))
        self.activity = copy.deepcopy(np.array(act.flat).T)

        rho_w = self.rho_w0*act[1:self.N-1,1:self.N-1, 1:self.N-1]
        lambda_w = self.lambda_w0*weight

        rho_wf = self.k*self.tau/self.w0*rho_w*c[1:self.N-1,1:self.N-1,1:self.N-1]
        lambda_wf = self.k*self.tau/self.w0*lambda_w[1:self.N-1,1:self.N-1,1:self.N-1]*v[1:self.N-1,1:self.N-1,1:self.N-1]*v_rad[1:self.N-1,1:self.N-1,1:self.N-1]
        
        w[1:self.N-1,1:self.N-1,1:self.N-1]= (self.a1*w[1:self.N-1,1:self.N-1,1:self.N-1] 
        + self.a2*w[2:self.N,1:self.N-1,1:self.N-1] 
        + self.a3*w[1:self.N-1,2:self.N,1:self.N-1]
        + self.a4*w[1:self.N-1,1:self.N-1,2:self.N] 
        + self.a5*w[0:self.N-2,1:self.N-1,1:self.N-1] 
        + self.a6*w[1:self.N-1,0:self.N-2,1:self.N-1]
        + self.a7*w[1:self.N-1,1:self.N-1,0:self.N-2]
        + rho_wf - lambda_wf)

        temp_w=np.array(w.flat)
        temp_w[boundary]=1
        # temp_n = np.reshape(temp_n,(variables.N, variables.N, variables.N)) 
        self.waste=copy.deepcopy(temp_w[0:].T)

    def TAFEq(self, v_rad, boundary, ux, uy, uz):
        """
        Calculates Angiogenesis factors within the environment 
        :param weight: 
        :param v_rad:
        :param boundary: boundary of the tumor environment 
        :param ux: the x component of the pressure gradient 
        :param uy: the y component of the pressure gradient
        :param uz: the z component of the pressure gradient
    """
        n = self.nutr.reshape(self.N, self.N, self.N)
        c = self.celltype.reshape(self.N,self.N, self.N)
        v = self.vess_tag.reshape(self.N, self.N, self.N)
        t = self.TAF.reshape(self.N,self.N, self.N)
        

        #### TAF Equation ####
        if self.grad == True:
            self.s1 = 1-6*self.Dc*self.tau*self.k/((self.L**2)*(self.h**2))
            self.s2 = self.Dc*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*ux[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h)
            self.s3 = self.Dc*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*uy[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.s4 = self.Dc*self.tau*self.k/((self.L**2)*(self.h**2))-self.tau*self.k*self.u0/self.L*uz[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.s5 = self.Dc*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*ux[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.s6 = self.Dc*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*uy[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h) 
            self.s7 = self.Dc*self.tau*self.k/((self.L**2)*(self.h**2))+self.tau*self.k*self.u0/self.L*uz[1:self.N-1,1:self.N-1,1:self.N-1]/(2*self.h)
            self.grad = False

        n_new=n[1:self.N-1,1:self.N-1,1:self.N-1]
        n_new[np.where(n_new > 1)]=1

        rho_c = self.rho_c0*(1-n_new)
        lambda_c = self.lambda_c0*v_rad

        rho_cf = self.k*self.tau/self.c0*rho_c*c[1:self.N-1,1:self.N-1,1:self.N-1]
        lambda_cf = self.k*self.tau/self.c0*lambda_c[1:self.N-1,1:self.N-1,1:self.N-1]*v[1:self.N-1,1:self.N-1,1:self.N-1]

        t[1:self.N-1,1:self.N-1,1:self.N-1]= (self.s1*t[1:self.N-1,1:self.N-1,1:self.N-1] 
        + self.s2*t[2:self.N,1:self.N-1,1:self.N-1] 
        + self.s3*t[1:self.N-1,2:self.N,1:self.N-1]
        + self.s4*t[1:self.N-1,1:self.N-1,2:self.N] 
        + self.s5*t[0:self.N-2,1:self.N-1,1:self.N-1] 
        + self.s6*t[1:self.N-1,0:self.N-2,1:self.N-1]
        + self.s7*t[1:self.N-1,1:self.N-1,0:self.N-2]
        + rho_cf - lambda_cf)

        t=np.array(t.flat)
        t[boundary]=0
        self.TAF = copy.deepcopy(t)
        

    def growth(self):
        """
            method used to set the tumor up to undergo cell division 
        """
        ##### Tumor Model Growth ######
        cellindex = np.array(np.where(self.celltype.flat > self.nec))
        test_act = self.activity[cellindex]
        test_cell_energy = self.cell_energy[cellindex]
        self.celltype[cellindex[np.where(test_act >= 0.5)]]=1                     # Active tumor cells
        self.celltype[cellindex[np.where(test_act < 0.5)]]=0.95                   # Quescient tumor cells
        self.celltype[cellindex[np.where(test_cell_energy <= 0)]]=self.nec        # Necrosis Cells
        
        # Cell Proliferation
        div_index = np.where(self.celltype.flat == .95)[0]                        # Randomizing the quiscent tumor cell order
        np.random.shuffle(div_index)
        for ss in range(0, div_index.size):
            s = div_index[ss]
            self.cell_energy[s] = self.cell_energy[s] - 0.2

        div_index = np.where(self.celltype.flat == 1)[0]
        for ss in range(0,div_index.size):
            s = div_index[ss]
            prolif_energy = 30

            if np.any(self.cell_energy[s] >= prolif_energy):
                self.Celldivide_anisotropic_anatomy_v3(s)
                #self.cellmove_v4(s)
            else:
                self.cell_energy[s] = self.cell_energy[s] + self.activity[s] - 0.65*(self.activity[s]/(self.activity[s]+1))
    
    
    def Angiogeneis3D(self,s):
        """
            Methods to calculate and produce Angiongeneis within
            the tumor environment
            :param s: the data point location in which Angiogenesis is 
                      being calculated
        """
        if 'count' not in self.vess[s]:
            self.vess[s]['count'] = 0
            
        if self.vess[s]['count'] >= (10*((1/10)**(1/30))**(30 - self.pres[s]*60)):
            gro = self.direction( s, self.index_bias)
            if np.any(gro):
                if self.vess_tag[0:,s] == 1:
                    self.vess[s]['son'] = s+self.index_bias[gro]
                    self.vess_tag[s+self.index_bias[gro]] = 0.95
                    if self.vess_age[s] > 1:
                        self.vess_age[s+self.index_bias[gro]] = 1
                    else:
                        self.vess_age[s+self.index_bias[gro]] = self.vess_age[s]/2
                    
                    self.vess[s+self.index_bias[gro]]['count'] = 0
                    self.vess[s+self.index_bias[gro]]['pare'] = s
                    self.vess[s+self.index_bias[gro]]['son'] = []
                    self.vess[s+self.index_bias[gro]]['direct'] = self.index_bias[gro]

                    self.branchrecord[s] = 1

                    self.sprout_index = np.setdiff1d(self.sprout_index,s)

                elif self.vess_tag[0:,s] == 0.95:

                    if self.hotpoint[0:,s] == 0:
                        self.vess[s]['son'] = s+self.index_bias[gro]
                        self.vess_tag[0:,s] = 1
                        self.vess_tag[0:,s+self.index_bias[gro]] = 0.95
                        if self.vess_age[0:,s] > 1:
                            self.vess_age[0:,s+self.index_bias[gro]] = 1
                        else:
                            self.vess_age[s+self.index_bias[gro]] = self.vess_age[s]/2
                        
                        self.vess[s+self.index_bias[gro]]['count'] = 0
                        self.vess[s+self.index_bias[gro]]['pare'] = s
                        self.vess[s+self.index_bias[gro]]['son'] = []
                        self.vess[s+self.index_bias[gro]]['direct'] = self.index_bias[gro]
            
                    else:
                        self.vess[s]['son'] = s+self.index_bias[gro]
                        self.vess_tag[0:,s] = 1
                        self.vess_tag[0:,s+self.index_bias[gro]] = 0.95
                        if self.vess_age[0:,s] > 1:
                            self.vess_age[0:,s+self.index_bias[gro]] = 1
                        else:
                            self.vess_age[s+self.index_bias[gro]] = self.vess_age[s]/2
                    
                        self.vess[s+self.index_bias[gro]]['count'] = 0
                        self.vess[s+self.index_bias[gro]]['pare'] = s
                        self.vess[s+self.index_bias[gro]]['son'] = []
                        self.vess[s+self.index_bias[gro]]['direct'] = self.index_bias[gro]
                        
                        index_bias2 = np.setdiff1d(self.index_bias,self.index_bias[gro])

                        gro = self.direction(s, index_bias2)

                        if np.any(gro):

                            self.vess[s]['son'] = s+index_bias2[gro]
                            self.vess_tag[s] = 1
                            
                            self.vess_tag[s+index_bias2[gro]] = 0.95
                            if self.vess_age[s] > 1:
                                self.vess_age[s+index_bias2[gro]] = 1
                            else:
                                self.vess_age[s+index_bias2[gro]] = self.vess_age[s]/2
                            self.vess[s+index_bias2[gro]]['count'] = 0
                            self.vess[s+index_bias2[gro]]['pare'] = s
                            self.vess[s+index_bias2[gro]]['son'] = []
                            self.vess[s+index_bias2[gro]]['direct'] = index_bias2[gro]
                        
                        self.branchrecord[s] = 1
            self.vess[s]['count'] = self.vess[s]['count']+1
        else:
            self.vess[s]['count'] = self.vess[s]['count']+1

    def direction(self, s, index_bias2):
        """
            This is method determines the direction in Angiogenesis will
            further progress
            :param s: the vessel data point of interest 
            :param index_bias2: the index bias for the vessel data points
            :returns: new point for which the angiogeneis will grow
        """
        nec = 1e-4
        direct_pare = self.vess[s]['direct']

        if not np.any(self.vess[s]['direct']):
            scalar = np.ones((index_bias2.size, 1))
        else:
            A,B,C = np.intersect1d((direct_pare.T+ [1, 1, self.N, -1*self.N, self.slen, -1*self.slen]),index_bias2, return_indices = True)
            scalar = np.zeros((index_bias2.size, 1))
            scalar[C] = 1
        
        prob0 = self.TAF[s+index_bias2] - np.tile(self.TAF[s], (index_bias2.size,1)).T
        prob = copy.copy(prob0)
        index1 = np.where(prob >= 0)
        index2 = np.where(prob < 0)
        prob[index1] = 1
        prob[index2] = 0

        prob = prob0 * prob

        scalar2 = (self.celltype[s+index_bias2] != self.nec).astype(int).T

        prob = prob.T * (self.vess_tag[0:, s+index_bias2]==0).T * scalar * scalar2

        gro = []

        if np.any(np.where(prob>0)):
            if random.random() < self.pres[s]:

                pres0 = self.pres[s+index_bias2].T - np.tile(self.pres[s], (index_bias2.size, 1))
                prob = pres0*(prob==1)
                prob = np.tile(norm(prob), (index_bias2.size, 1)) * (prob!=0) - prob
            
            prob = prob/norm(prob)
            prob = prob/np.sum(prob)
            prob = np.cumsum(prob)

            prob = np.insert(prob, 0,0)
            mov = random.random()

            for i in range(0, index_bias2.size):
                if mov == 0:
                    gro = 1
                    return gro
                else:
                    if mov>prob[i]:
                        if mov<=prob[i+1]:
                            gro = i
                            return gro
        else:
            gro = []
            return gro


    def iter(self, value, s, index):
        """
            This method intializes the vessel variable
            :param value: range of values being initialized
            :param s: the data pointof interest 
            :param index: a variable that dictates which direction the blood vessel
                          can grow from the specified data point 
            :returns: vessel dictionary variable
        """
        for i in value:
            self.vess[int(i)] = {'count': 0, 'pare': s, 'son': [], 'direct': index}



    def Spreadhotpoint(self, magnitude):
        """
            Method to set up hot points, branch points for vessels
            :param magnitude: constant variable that indicates the which
                              values allow a data point to be a hotpoint
        """
        self.hotpoint = np.zeros((1,self.wlen))

        posi = magnitude*1.3 ** np.log(self.TAF)
        index = np.where(np.random.rand(self.wlen,1).T<posi)
        self.hotpoint[index] = 1

    def SproutCheck(self):
        """
            Method to check for hotpoints, branching points for blood vessles
            :returns: an array of different potential branching points 
        """
        sprout_index = []
        vess_index = np.where(self.vess_tag==1)

        vess_index = np.setdiff1d(vess_index, np.where(self.branchrecord==1))
        return vess_index[np.where(self.hotpoint[0:,vess_index] == 1)[0]]
    def Angiogenesis(self, iteration):
        """
            Method to set up blood vessels to undergo Angiogenesis
            :param iteration: Determines the stating point for Angiogenesis to start
                              occuring
        """
        if iteration >= self.starttime:

            self.vessgrowth_flag = 1

            if iteration==self.starttime:
                tip_index = self.vindexsave

            self.vess_age[np.where(self.vess_tag>0)] = self.vess_age[np.where(self.vess_tag>0)] + 1
            tip_index = np.union1d(np.where(self.vess_tag==0.95)[1], self.sprout_index)
            np.random.shuffle(tip_index)

            for i in range(0,tip_index.size):
                s = int(tip_index[i])
                self.Angiogeneis3D(s)

            self.Spreadhotpoint(0.3e-3)
            self.sprout_index = self.SproutCheck()

    def anatomy_crop(self, array, x, y, z):
        range = int(self.N/2)
        arr = copy.deepcopy(array[x-range:x+range+1, y-range:y+range+1, z-range:z+range+1])
        raw_pathway = "{:s}/{:d}/cropped_anatomy.raw".format(
            self.output, self.seed, self.seed)
        fID = open(raw_pathway, 'w')
        arr1 = np.array(arr, dtype="uint8")
        arr1.tofile(raw_pathway)
        fID.close()
        return arr


    def main(self):
        """
            Main function of the class
        """
        start_time = time.time()
        while(len(self.coord) > 0):
            if self.loc_it != 0:
                self.re_init()
            self.loc_it += 1
            print("Starting Location {:d}".format(self.loc_it))
            coord = self.coord.pop(0)
            coord_x = int(coord[0])
            coord_y = int(coord[1])
            coord_z = int(coord[2])

            #Could potentially call cropped
            self.crop_anatomy = self.anatomy_crop(self.full_anatomy,coord_x,coord_z,coord_y)
            self.anatomy = "{:s}/{:d}/cropped_anatomy.raw".format(
                            self.output, self.seed)
            if np.size(self.crop_anatomy) < (self.N * self.N * self.N):
                print("Skipping this location: Array is too small", file=sys.stderr)
                continue
            boundary = DetectBoundary(self.nod3xyz, self.leng)
            self.InitCancerCell()
            self.ReadAnatomyData(self.crop_anatomy)
            t1 = np.setdiff1d(np.array(range(1,self.wlen)),boundary)

            calit = (self.time_array[-1])+2
            iter = self.time_array.pop(0)
            
            for iteration in range(0, calit):
                print("Interations: {:d}".format(iteration), flush=True)
                v_age = self.vess_age.reshape(self.N, self.N, self.N)
                v_rad = v_age/(self.k_AR2+v_age)
                if (iteration)%20 == 0:
                    self.pres = np.zeros(self.wlen)
                    cindex = np.array(np.where(self.celltype != 0))
                    vindex = np.array(np.where(self.vess_tag.flat > 0))
                    
                    self.CTP(cindex)
                    self.VTP(vindex)

                    self.pres = self.pres/self.p0
                    ux,uy,uz = self.Gradient(t1) 
                    self.grad = True

                p = np.reshape(self.pres,(self.N, self.N, self.N))
                weight = (self.cap_pres - p *self.p0)/ self.cap_pres
                weight[np.where(weight< 0)] = 0
                
                self.Nutrients(weight, v_rad, boundary, ux, uy, uz)
                self.Metabolic(weight,v_rad,boundary,ux,uy,uz)
                self.TAFEq(v_rad, boundary, ux, uy, uz)
                self.weight_old = np.sum(weight)
                    
                self.growth()
                self.Angiogenesis(iteration)

                if iteration == iter:
                    print("Iteration: {:d}".format(iteration))
                    with open("{:s}/time.txt".format(self.output), 'w') as f:
                        f.write("Iteration: {:d} Time: {:d} seconds \n".format(iteration, int(time.time() - start_time)))

                    write.writetoraw_portable(self.celltype, self.ana_no, self.mass_dim, self.N, self.wlen, iter,self.anatomy,self.output, self.seed, self.loc_it)
                    write.write_to_fullanatomy(self.celltype.reshape(self.N,self.N,self.N), self.ana_no, coord_x,coord_y,coord_z, iter, self.full_anatomy, self.N, self.output, self.seed, self.loc_it)
                    if len(self.time_array) > 0:
                        iter = self.time_array.pop(0)
                """
                if iteration > 249:
                    if iteration < 374:
                        if (iteration)%7 == 0:
                            self.days_forAndrea = round((iteration)/7)
                            ana_data = write.writetoraw_portable(self.celltype, self.ana_no, self.mass_dim, self.N, self.wlen, self.days_forAndrea,self.anatomy)
                            write.write_to_fullanatomy(ana_data, self.ana_no, self.coord_x,self.coord_y,self.coord_z, self.days_forAndrea, self.full_anatomy, self.N)
                    else:
                        if (iteration)%4 == 0:
                            self.days_forAndrea = round((iteration)/4)
                            ana_data = write.writetoraw_portable(self.celltype, self.ana_no, self.mass_dim, self.N, self.wlen, self.days_forAndrea,self.anatomy)
                            write.write_to_fullanatomy(ana_data, self.ana_no, self.coord_x,self.coord_y,self.coord_z, self.days_forAndrea, self.full_anatomy, self.N)
                """
        # self.mhd_read.write_mhd(self.mhd_data, self.output, self.seed)
        

            
"""
start_time = time.time()
ts = TumorSim('model_1', 201, [5,6,7,8,9,10],5)

cProfile.run('ts.main()', 'output.dat')

with open('output_time.txt', 'w') as f:
    p = pstats.Stats('output.dat', stream=f)
    p.sort_stats('time').print_stats()

with open('output_calls.txt', 'w') as f:
    p = pstats.Stats('output.dat', stream=f)
    p.sort_stats('calls').print_stats()


#ts.main()
#print("--- %s seconds ---" % (time.time() - start_time))
"""



