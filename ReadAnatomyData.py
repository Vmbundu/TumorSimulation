import numpy as np
import copy
from scipy.ndimage import gaussian_filter
import variables
global wlen 
global local_data
global N
global days
global ana_no
global mass_dim
def ReadAnatomyData():
    fileID = open('Input_parameters.txt', 'r')
    A = fileID.readlines()
    variables.days = int(A[0].rstrip());
    variables.ana_no = int(A[1].rstrip());
    variables.mass_dim = int(A[2].rstrip());


    atmy_filepath = 'AnatomyData/fatty_bckgrnd' + str(variables.ana_no) + '.raw'
    fID = open(atmy_filepath, 'r')
    #Make sure to implements global variables when I complete main function
    ana = np.fromfile(fID, dtype=np.uint16, count = 201 * 201 * 201, sep="")
    fID.close()

    data = copy.deepcopy(ana)
    np.place(ana, ana != 88, 1)
    #Similiar case for this file as well 
    # Noted: Diffcult to gauge results with that of the one in MATLAB, determining 
    # a workaround
    ana_re = ana.reshape(201,201,201)

    ana_smooth = gaussian_filter(ana_re, 0.45)
    ligan = np.where(ana_smooth.flat>12)
    for elem in ligan:
        data[elem] = 88

    atmy_thickliga_filepath = 'Results_Raw/PostProcessed_Anatomy/anatomy' + str(variables.ana_no) + 'thickliga.raw'
    fID = open(atmy_thickliga_filepath, 'w')
    data1 = np.array(data, dtype="uint8")
    data1.tofile(atmy_thickliga_filepath)
    #data = data.astype(np.uint8)
    #np.savetxt(atmy_thickliga_filepath, data, delimiter=",")
    #fID.close()

    datas = copy.deepcopy(data).astype(np.uint64)
    #np.place(datas, data==1, 10)
    datas[np.where(data==1)] = 10
    #np.place(datas, data==29,30)
    datas[np.where(data==29)] = 45
    #np.place(datas, data==88, 1e6)
    datas[np.where(data==88)] = 1e6
    #np.place(datas, data==125, 1)
    datas[np.where(data==125)] = 1
    #np.place(datas, data==95, 1e6)
    datas[np.where(data==95)] = 1e6
    #np.place(datas, data==150, 1e6)
    datas[np.where(data==150)] = 1e6
    #np.place(datas, data==225, 1e6)
    datas[np.where(data==225)] = 1e6
    #cond = [data != 1, data != 29, data != 88, data != 125, data != 95, data != 150, data != 225]
    cond = [1,2,29,88,125,95,150,225]
    datas[np.where((data != 1) & (data != 29) & (data != 88) & (data != 125) & (data != 95) & (data != 150) & (data != 225))] = 1

    #datas = np.reshape(datas, (variables.N, variables.N, variables.N))
    #datas[0:,0:,0:] = 1
    #datas[0:,0:, 110:200] = 100

    #datas = np.reshape(datas,(variables.N*variables.N*variables.N,))
    variables.local_data = datas.transpose()
