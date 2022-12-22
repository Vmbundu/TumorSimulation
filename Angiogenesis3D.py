import variables
import numpy as np
def Angiogenesis3D(s):
    if variables.vess[s].count >=(10*((1/10)^(1/30))^(30-variables.pres[s]*60)):
        gro = direction(s,index_bias)
        if np.any(gro):
            if variables.vess_tag[s] == 1:
                variables.vess[s].son = s+variables.index_bias[gro]

                if variables.vess_age[s] > 1:
                    variables.vess_age[s+ variables.index_bias]
                else:
                    variables.vess_age[s + variables.index_bias] = variables.vess_age[s]/2
                variables.vess[s+variables.index_bias[gro]].count = 0            # New endothelial cell: count = 0; count++ in each iteration
                variables.vess[s+variables.index_bias[gro]].pare = s             # Add new EC parent location
                variables.vess[s+variables.index_bias[gro]].son = []             # Add new EC son location, [] here
                variables.vess[s + variables.index_bias[gro]].direct = variables.index_bias[gro]
            else:   # Touching the branching hotpoint, EC starts to branch

                variables.vess[s].son = variables.index_bias[gro]               # vess{s}.son=[] when it was created, now it has one descendant, I add the son location.
                variables.vess_tag[s] = 1
                variables.vess_tag[s + variables.index_bias[gro]] = 0.95
                if variables.vess_age[s] > 1:
                    variables.vess_age[s + variables.index_bias[gro]] = 1       # New endothelial cell :  cellage=0; cellage++ in each iteration.
                else:
                    variables.vess_age[s + variables.index_bias[gro]] = variables.vess_age[s]/2
                
                variables.vess[s+variables.index_bias[gro]].count=0    # New endothelial cell :  count=0; count++ in each iteration.
                variables.vess[s+variables.index_bias[gro]].pare=s     # Add new EC parent location.
                variables.vess[s+variables.index_bias[gro]].son=[]     # Add new EC son location, [] here. 
                variables.vess[s+variables.index_bias[gro]].direct=variables.index_bias[gro]

                 



