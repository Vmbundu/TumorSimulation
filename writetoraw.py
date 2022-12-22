import variables
import numpy as np
import os
def writetoraw():
    tumorex = variables.celltype
    tumorex[np.where(tumorex>0)] = 1
    tumorex[np.where(tumorex<=0)] = 0

    ana_mno = variables.ana_mo%10
    mid_dim = round(variables.mass_dim/2)

    SD = 4
    d = variables.days

    subfold1 = '/Results_Raw/SisterneModels'
    subfold2 = '/Results_Raw/ForInsertion'
    subfold3 = '/Results_Raw/Lesion'
    subfold4 = '/Results_Raw/Lesion_wAnatomy'
    subfold5 = '/Results_Raw/PostProcessed_Anatomy'
    if os.path.exists(subfold1) == False:
        os.makedirs(subfold1)
    if os.path.exists(subfold2) == False:
        os.makedirs(subfold2)
    if os.path.exists(subfold3) == False:
        os.makedirs(subfold3)
    if os.path.exists(subfold4) == False:
        os.makedirs(subfold4)
    if os.path.exists(subfold5) == False:
        os.makedirs(subfold5)

    sis_filepath = 'Results_Raw/SisterneModels/mass_' + str(variables.ana_no) + '_' + str(variables.mass_dim) + '_d' + str(d) +'.raw'
    fID = open(sis_filepath, 'r')
    ana = np.fromfile(fID, dtype=np.uint8, count = variables.mass_dim * variables.mass_dim * variables.mass_dim, sep="")
    fID.close()

    ana = np.reshape(ana,(variables.mass_dim, variables.mass_dim, variables.mass_dim))
    ana_l = np.zeros((variables.mass_dim, variables.mass_dim, variables.mass_dim))
    tmr_re = np.reshape(tumorex,(variables.N, variables.N, variables.N))

    for x in range(-100, 101):
        for y in range(-100,101):
            for z in range(-100,101):
                if tmr_re[x+101,y+101,z+101] == 1:
                    ana_l[x+mid_dim,y+mid_dim,z+mid_dim] = 1
    
    spic = np.where(ana == 2)[0]
    ana_l[spic] = 1

    ### For generating Raw Files for Lesion Insertion ####
    savfile = 'Results_Raw/ForInsertion/tumor_' + str(variables.ana_no) + '_SD' + str(SD) + '_d' + str(d) +'.raw'
    fID = open(savfile, 'w')
    kkk_tumor = np.reshape(ana_l,(variables.mass_dim, variables.mass_dim, variables.mass_dim))
    kkk_tumor = np.array(kkk_tumor, dtype="uint8")
    ## Check on Permute Method
    kkk_tumor.tofile(savfile)

    ####################################
    les_filepath = 'Results_Raw/Lesion/tumor_' + str(variables.ana_no) + '_SD' + str(SD) + '_d' + str(d) +'.raw'

    fID = open(les_filepath,'w')
    kkk_tumor = np.reshape(tumorex,(variables.N, variables.N, variables.N))
    kkk_tumor = np.array(kkk_tumor, dtype="uint8")
    kkk_tumor.tofile(les_filepath) 
    fID.close

    les_wana_filepath = 'Results_Raw/Lesion_wAnatomy/tumor_wAnatomy' + str(variables.ana_no) + '_SD' + str(SD) + '_d' + str(d) +'.raw'
    atmy_thickliga_filepath = 'Results_Raw/PostProcessed_Anatomy/anatomy' + str(variables.ana_no) + '_thickliga.raw'
    fID = open(atmy_thickliga_filepath,'r')
    data = np.fromfile(fID, dtype=np.uint8, count = variables.wlen, sep="")
    datas = data.transpose
    les_pos = np.where(tumorex == 1)
    datas[les_pos] = 50
    fID.close

    fID = open(les_wana_filepath, 'w')
    kkk_tumor = np.reshape(datas, (variables.N, variables.N, variables.N))
    kkk_tumor = np.array(kkk_tumor, dtype="uint8")
    kkk_tumor.tofile(les_wana_filepath) 
    fID.close



