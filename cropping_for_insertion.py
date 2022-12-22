import variables
import numpy as np
def cropping_for_insertion():
    
    les_filepath = 'Results_Raw/ForInsertion/tumor_' + str(variables.ana_no) + '_SD4' + '_d' + str(variables.days) +'.raw'
    fID = open(les_filepath,'r')
    les = np.fromfile(fID, dtype=np.uint8, count = variables.mass_dim*variables.mass_dim*variables.mass_dim, sep="")
    fID.close

    les_s = np.zeros((variables.N, variables.N, variables.N))
    les = np.reshape(les,(variables.mass_dim, variables.mass_dim, variables.mass_dim))
    mid_dim = round(variables.mass_dim/2)

    for x in range(-100, 101):
        for y in range(-100,101):
            for z in range(-100,101):
                if les[x+mid_dim,y+mid_dim,z+mid_dim] == 1:
                    les_s[x+101, y+101, z+101] = 1
    les_filepath = 'Results_Raw/ForInsertion/tumor_cropped_' + str(variables.ana_no) + '_SD4' + '_d' + str(variables.ana_no) +'.raw'
    fID = open(les_filepath, 'w')
    kkk_tumor = np.reshape(les_s,(variables.N, variables.N, variables.N))
    kkk_tumor = np.array(kkk_tumor, dtype="uint8")
    kkk_tumor.tofile(les_filepath)
    fID.close

