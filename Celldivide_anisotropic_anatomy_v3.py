import variables
import random
import numpy as np
import math
def Celldivide_anisotropic_anatomy_v3(s):
    # variables.init()
    variables.stackcount = 0
    variables.nec = 1e-4
    prob = variables.celltype[0,s+variables.index_bias]
    index1 = np.where(prob > 0)
    index2 = np.where(prob == 0)

    # Generating an array that has 1 to indicate empty positions

    prob[index1] = 0
    prob[index2] = 1

    prob1 = variables.local_data[s+variables.index_bias]

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
        prob = np.ones(len(variables.index_bias))
        scalar0 = (variables.celltype[0,s + variables.index_bias] != variables.nec)
        prob = prob * scalar0

        if np.any(np.where(prob == 1)): # when are at least some locations which are non-necrotic
            pres_loc = variables.pres[0,s+variables.index_bias] # calculated pressure
            pres_dir = variables.local_data[s+variables.index_bias] # scalar multiplicative factors corresponding to each tissue type
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

            if (variables.celltype[0,s + variables.index_bias[gro]] == 0): # if the randomly chosen location doesn't have a cell
                variables.celltype[0,s + variables.index_bias[gro]] = 1
                variables.cell_energy[0,s] = 15
            else:
                variables.celltype[0,s + variables.index_bias[gro]] = 1 # if the randomly chosen location already has a cell, then the cells need to move
                variables.cell_energy[0,s] = 15
                old_cell_energy = variables.cell_energy[0, s + variables.index_bias[gro]]
                variables.cell_energy[0, s +  variables.index_bias[gro]] = 15
                variables.stackvalue = [ s + variables.index_bias[gro], old_cell_energy]
                cellmove_v3()
    else:   # there are several empty locations, which are not ligaments or stiffer  tissues, to add cells
        pres_loc = variables.pres[0,s + variables.index_bias] # calculated pressure
        pres_dir = variables.local_data[s + variables.index_bias] # scalar multiplicative factors corresponding to each tissue type
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
        variables.celltype[0,s + variables.index_bias[gro]] = 1
        variables.cell_energy[0,s] = 15
        variables.cell_energy[0,s + variables.index_bias[gro]] = 15

def cellmove_v3(): # Cells are pushing to move outward
    # This function is used for tumor cell movement when inner cell proliferation
    variables.stackcount = variables.stackcount+1

    nec = 1e-4

    if variables.stackcount <= 900:

        s = variables.stackvalue[0]
        old_cell_energy = variables.stackvalue[1]

        prob = variables.celltype[0,s+variables.index_bias]
        index1 = np.where(prob > 0)
        index2 = np.where(prob == 0)

        prob[np.array(index1)] = 0
        prob[np.array(index2)] = 1

        prob1 = variables.local_data[s + variables.index_bias]
        index1 = np.where(prob1 > 50)
        index2 = np.where(prob1 <= 50)

        prob1[index1] = 0
        prob1[index2] = 1

        prob2 = prob * prob1

        if not np.any(np.where(prob2 == 1)):
            prob = np.ones(len(variables.index_bias))
            scalar0 = (variables.celltype[0,s + variables.index_bias] != variables.nec)
            prob = prob * scalar0
        
        if np.any(np.where(prob == 1)):

            pres_loc = variables.pres[0,s + variables.index_bias]
            pres_dir = variables.local_data[s + variables.index_bias]
            pres_var = pres_loc * pres_dir
            pres_var[np.where(pres_var == 0)] = 1
            pres_prob = 1/pres_var
            #pres_prob[np.where(pres_var != 0)] = 1/pres_var[np.where(pres_var != 0)]
            #if np.any(pres_var == 0):
            #    pres_prob[np.where(pres_var == 0)] = 1
            

            #pres_inf = np.where(pres_prob == math.inf)

            #if np.any(pres_inf):
            #    pres_prob[pres_inf] = 1
            #    stop = 1
            
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

            if variables.celltype[0,s + variables.index_bias[gro]] == 0:
                variables.celltype[0,s + variables.index_bias[gro]] = 1
                variables.cell_energy[0,s + variables.index_bias[gro]] = old_cell_energy
            else:
                variables.celltype[0,s + variables.index_bias[gro]] = 1
                old_cell_energy_move = variables.cell_energy[0,s + variables.index_bias[gro]]
                variables.cell_energy[0,s + variables.index_bias[gro]] = old_cell_energy
                variables.stackvalue = [s + variables.index_bias[gro], old_cell_energy_move]
                cellmove_v3()


