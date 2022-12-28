import variables
import copy
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
def writeraw_forAndrea():
    tumorex = copy.deepcopy(variables.celltype)
    
    tumorex[np.where(tumorex>0)] = 1
    tumorex[np.where(tumorex<=0)] = 0

    ana_mno = variables.ana_no % 10
    mid_dim = round(variables.mass_dim/2)

    SD = 4
    d = variables.days_forAndrea
    subfold1 = 'Results_RawforAndrea/Lesion'
    subfold2 = 'Results_RawforAndrea/Lesion_wAnatomy'
    subfold3 =  'Results_Raw/PostProcessed_Anatomy'

    if os.path.exists(subfold1) == False:
        os.makedirs(subfold1)
    if os.path.exists(subfold2) == False:
        os.makedirs(subfold2)
    if os.path.exists(subfold3) == False:
        os.makedirs(subfold3) 

    les_filepath = 'Results_RawforAndrea/Lesion/tumor_' + str(variables.ana_no) + '_SD' + str(SD) + '_d' + str(d) +'.raw'
    fID = open(les_filepath, 'w')
    kkk_tumor = np.reshape(tumorex,(variables.N, variables.N, variables.N))
    kkk_tumor = np.array(kkk_tumor, dtype="uint8")
    kkk_tumor.tofile(les_filepath)
    fID.close()

    les_wana_filepath = 'Results_RawforAndrea/Lesion_wAnatomy/tumor_wAndrea' + str(variables.ana_no) + '_SD' + str(SD) + '_d' + str(d) +'.raw'
    atmy_thickliga_filepath = 'Results_Raw/PostProcessed_Anatomy/anatomy' + str(variables.ana_no) + '_thickliga.raw'
    fID = open(atmy_thickliga_filepath, 'r')
    data = np.fromfile(fID, dtype=np.uint8, count = variables.wlen, sep="")
    datas = data.transpose()
    les_pos = np.where(tumorex == 1)[0]
    datas[les_pos] = 50
    fID.close

    fID = open(les_wana_filepath, 'w')
    kkk_tumor = np.reshape(datas,(variables.N, variables.N, variables.N))
    kkk_tumor = np.array(kkk_tumor, dtype="uint8")
    kkk_tumor.tofile(les_wana_filepath)
    fID.close

def writetoraw_portable(celltype, ana_no, mass_dim, N, wlen, days_forAndrea, anatomy, results_folder, seed,loc_it):
    """
        Method to write the data into raw files for the direct tumor environment 
        :param celltype: celltype data
        :param ana_no:
        :param mass_dim:
        :param N: size of the tumor environment 
        :param wlen: N * N * N
        :param days_forAndrea: days that have elapsed for the tumor
        :param anatomy: the anatomy file
    """
    tumorex = copy.deepcopy(celltype)
    
    tumorex[np.where(tumorex>0)] = 1
    tumorex[np.where(tumorex<=0)] = 0

    print('Cell Size:'+ str(np.sum(tumorex)))
    ana_mno = ana_no % 10
    mid_dim = round(mass_dim/2)

    SD = 4
    d = days_forAndrea
    """
    subfold1 = 'Results_RawforAndrea/Lesion'
    subfold2 = 'Results_RawforAndrea/Lesion_wAnatomy'
    subfold3 =  'Results_Raw/PostProcessed_Anatomy'
   

    if os.path.exists(subfold1) == False:
        os.makedirs(subfold1)
    if os.path.exists(subfold2) == False:
        os.makedirs(subfold2)
    if os.path.exists(subfold3) == False:
        os.makedirs(subfold3) 
    
    """
    subfold = '{:s}/loc_{:d}'.format(results_folder,loc_it)
    if os.path.exists(subfold) == False:
        os.makedirs(subfold)
    #les_filepath = 'Results_RawforAndrea/Lesion/tumor_' + str(ana_no) + '_SD' + str(SD) + '_d' + str(d) +'.raw'
    #fID = open(les_filepath, 'w')
    kkk_tumor = np.reshape(tumorex,(N, N, N))
    kkk_tumor = np.array(kkk_tumor, dtype="uint8")
    with h5py.File('{:s}/loc_{:d}/pcl_{:d}_resTumor.hdf5'.format(results_folder, loc_it, seed), 'a') as f:
        if 'test_'+str(d) in f:
            del f['test_'+str(d)]
        f.create_dataset('test_'+str(d), data = kkk_tumor, compression='gzip')
    #kkk_tumor.tofile(les_filepath)
    #fID.close()

    #les_wana_filepath = 'Results_RawforAndrea/Lesion_wAnatomy/tumor_wAndrea' + str(ana_no) + '_SD' + str(SD) + '_d' + str(d) +'.raw'
    #atmy_thickliga_filepath = 'Results_Raw/PostProcessed_Anatomy/anatomy' + str(ana_no) + '_thickliga.raw'
    atmy_thickliga_filepath = anatomy
    fID = open(atmy_thickliga_filepath, 'r')
    data = np.fromfile(fID, dtype=np.uint8, count = wlen, sep="")
    datas = data.transpose()
    les_pos = np.where(tumorex == 1)[0]
    datas[les_pos] = 50
    fID.close

    #fID = open(les_wana_filepath, 'w')
    #kkk_tumor = np.reshape(datas,(N, N, N))
    data = np.reshape(datas,(N,N,N))
    #kkk_tumor = np.array(kkk_tumor, dtype="uint8")
    data = np.array(data, dtype="uint8")
    tumorex = np.reshape(tumorex,(N,N,N))
    data[np.where(tumorex == 1)] = 200
    #kkk_tumor.tofile(les_wana_filepath)
    #data.tofile(les_wana_filepath)
    #fID.close
    with h5py.File("{:s}/loc_{:d}/pcl_{:d}_res.hdf5".format(results_folder, loc_it, seed), 'a') as f:
        if 'test_'+str(d) in f:
            del f['test_'+str(d)]
        f.create_dataset('test_'+str(d), data = data, compression='gzip')
    return data

def write_to_fullanatomy(data, ana_no, x,y,z, days_forAndrea, full_anatomy, size, results_folder, seed, loc_it):
    """
        Method to write the data into raw files for the full anatomy
        :param data: tumor data points
        :param ana_no:
        :param x: x coordinate of the tumor within the full anatomy 
        :param y: y coordinate of the tumor within the full anatomy
        :param z: z coordinate of the tumor within the full anatomy
        :param days_forAndrea: days that have elapsed for the tumor
        :param full_anatomy: data points for the full anatomy
        :param size: size of the direct tumor environment

    subfold4 = 'Results_RawforAndrea/Lesion_wFullAnatomy'

    if os.path.exists(subfold4) == False:
        os.makedirs(subfold4)
    """
    subfold = '{:s}/loc_{:d}'.format(results_folder,loc_it)
    if os.path.exists(subfold) == False:
        os.makedirs(subfold)
    SD = 4
    d = days_forAndrea

    # les_wana_filepath = 'Results_RawforAndrea/Lesion_wFullAnatomy/tumor_wAndrea_wfullAnatomy_' + str(ana_no) + '_SD' + str(SD) + '_d' + str(d) +'.raw'
    range = int(size/2)
    full_anatomy[x - range:x+range+1, z - range:z+range+1, y - range: y+range+1] = data
    #fID = open(les_wana_filepath, 'w')
    #full_anatomy.tofile(les_wana_filepath)
    #fID.close()
    plt.imshow(full_anatomy[0:,0:,200])
    plt.savefig("fullAna.jpg")
    with h5py.File("{:s}/loc_{:d}/pcl_{:d}_resFull.hdf5".format(results_folder, loc_it, seed), 'a') as f:
        if 'test_'+str(d) in f:
            del f['test_'+str(d)]
        f.create_dataset('test_'+str(d), data = full_anatomy, compression='gzip')