import os
import numpy as np
import Celldivide_anisotropic_anatomy_v3 as cell
import DetectBoundary as db
import InitCancerCell as init
import variables
import matplotlib.pyplot as plt
import math
import ReadAnatomyData as anatomy
import Vess
import copy 
import Spreadhotpoint as spread
import writetoraw_forAndrea as write
from Bias import Bias
from Pres import Pres

#===================Multiscale Tumor Growth and Angiogenesis Model===========
#
# This is the main function to run the simulation.
#===============Start=================
#Create  a folder to save computational results
print('================ Simulate Tumor Growth and Angiogenesis ===================')
subfold1 = 'Results_Matlab/Results_Matlab/Figures'
subfold2 = 'Results_Matlab/Results_Matlab/Data'
if os.path.exists(subfold1) == False:
    os.makedirs(subfold1)
if os.path.exists(subfold2) == False:
    os.makedirs(subfold2)
variables.init()


del subfold1
del subfold2


boundary = db.DetectBoundary(variables.nod3xyz,variables.leng)
init.InitCancerCell()
plt.show()
anatomy.ReadAnatomyData()
#============== Initial vessel cells ============
# Initial sprouting points located along artificial vasculature 
X = [100, 135, 172, 183, 187, 181, 157, 123, 86, 52, 24, 19, 45, 18, 19]
Y = [186, 179, 163, 123, 87, 47, 25, 15, 14, 23, 44, 148, 180, 77, 115]
Z = np.array([101]*15)

varalist = ['TAF', 'activenumber', 'act', 'activity', 'cell_energy', 'celltype',
            'necrosisnumber', 'quiescentnumber', 'pres', 'nutr', 'v_age', 'vess',
            'vess_age', 'vess_tag', 'waste']
vindex = np.zeros(len(X), dtype=int)

for i in range(0, len(X)):
    xx = round((X[i]/20)/0.05)
    yy = round((Y[i]/20)/0.05)
    zz = round((Z[i]/20)/0.05)

    vindex[i] = math.trunc(xx+yy*variables.N+zz*variables.slen)
ax = plt.axes(projection="3d")
ax.scatter3D(variables.nod3xyz[vindex,0]*20,variables.nod3xyz[vindex,1]*20, variables.nod3xyz[vindex,2]*20)

for j in range(0,vindex.size):
    s = vindex[j]
    variables.vess[s] = Vess.Vess(s)

variables.vess = np.array(variables.vess)
for i in vindex:
    variables.vess_tag[0][i] = 0.95
    variables.vess_age[0][i] = 1

vindexsave = vindex
old_vindexsave = vindex

del vindex
del i 
del X
del Y 
del Z
del xx
del yy
del zz

totDays = 60 # Total simualation days (two months)
tau = totDays*24*3600 # Unit: second
L = 1e-2 # Unit: m, computational domain length: 10 mm
k = 1/(variables.leng*(variables.N-1)) # Normalized time step = 0.0005. 2000 interation
h = 1/(variables.N-1) # Spatial step size for numerical method - central difference is used for convection term

c0 = 4.3e-4 # Standard TAF concentration, unit: m^2/s [Wang & Li, 1998]           Original: 8e-14
n0 = 8.4    # Standard nutrient concentration, unit: mol/m^3,  || Refer to "HEMOGLOBIN-BASED OXYGEN CARRIER HEMODILUTION AND BRAIN OXYGENATION"
w0 = 10.5   # Nutrient consumption rate [Vaupel et al, 1987]

# Waste Parameters
Dn = 8e-14 # Nutrient diffusion rate, unit: m^2/s [Wang & Li, 1998]              %Origial : 8e-14
rho_n0      = 6.8e-4  # Vessel nutrient supply rate, unit: mol/(m^3*sec) [Wang & Li, 1998] %Origial : 6.8e-4
lambda_n0   = 3.0e-5  # Nutrient consumption rate [Vaupel et al, 1987]

# Waste parameters
Dw         = 4e-14  # Carbon dioxide diffusion coefficient, unit: m^2/s [estimated] %% Original: 4e-14
rho_w0     = 1e-5  # Carbon dioxide secretion rate, unit: mol/(m^3s) [estimated]    %% Original: 1e-5
lambda_w0  = 2.5e-5  # Carbon dioxide consumption rate, unit: ml/(cm^3s) [estimated]%% Original: 2.5e-5

# TAF parameters
Dc          = 1.2e-13  # TAF diffusion coefficient [estimated] 
rho_c0      = 2e-9  # TAF secretion rate [estimated]
lambda_c0   = 0  # VEGF natural decay rate, assumed to be very small [estimated]

# Pressure
pres_scale  = 1  # Tumor intersitial presssure: 1~60 mmHg, by adjusting this parameter, I can therefore investigate the growth patterns of different types of tumor, low pressure, high pressure.
                 # Our hypothesis is that the interstitial pressure inside solid tumor will give rise to different morphologies. Bigger interstitial pressure gives rise to dendritic tumor.
cap_pres    = 30  # unit: mmHg
p0          = 60*pres_scale  # unit: mmHg %% Change made by Aunnasha: Using 40 for round tumors as compared to 60 for dendritic tumors

Kp          = 4.5e-15  # Hydraulic conductivity of the interstitium, unit: cm^2/mmHg-sec
u0          = 5e-6
pv          = cap_pres/p0  # Capillary/vascular pressure, unit: (20) mmHg 
sigma0      = 0.15
amplitude0  = 0.08*pres_scale  # Gaussian function amplitude

k_AR2       = 500

############ index_bias is used to evaluate tumor cell density ################
#kn=1
#index_bias0=[]
#seq = range(-kn,kn+1)
#for i in range(-kn,kn+1):
#    index_bias0 = [*index_bias0, *(np.array(seq) + variables.N*i )]
#for j in range(-kn,kn+1):
#    for ele in index_bias0:
#        variables.index_bias.append(ele+variables.slen*j)

#variables.index_bias = np.array(variables.index_bias)

bias = Bias(variables.N, variables.slen)
variables.index_bias = bias.index_bias()
############# pres_bias is used to calculate tumor pressure ############
#knn = 20     ## to include influence of tumor cells upto 1 mm away
#pres_bias0 = []
#pres_bias = []
#seq = range(-knn,knn+1)

#for i in range(-knn,knn+1):
#    pres_bias0 = [*pres_bias0, *(np.array(seq) + variables.N*i)]
#for j in range(-knn,knn+1):
#    for ele in pres_bias0:
#        pres_bias.append(ele+variables.slen*j)
#pres_bias = np.array(pres_bias)

pres_bias = bias.pres_bias()


t1 = np.setdiff1d(np.array(range(1,variables.wlen)),boundary)

# Uncomment these two lines if you want to start the simulation from a
# certain day, e.q., day 20th. But for this, you need to have an input of 
# simulation at day 20. %% Don't use this : Not accurate : Aunnasha Aug 27
# 2020
#  filename=['./Results_Matlab/Data'];
#  load([filename,'/DataDay' num2str(14) '.mat'])

# 1 day = 33 iterations
# 40 days = 1321 iterations
# 60 days = 1981 iterations

calit = (25*variables.days)

for iteration in range(0,calit):

    t = variables.TAF.reshape(variables.N,variables.N, variables.N)
    n = variables.nutr.reshape(variables.N, variables.N, variables.N)
    v = variables.vess_tag.reshape(variables.N, variables.N, variables.N)
    c = variables.celltype.reshape(variables.N,variables.N, variables.N)
    w = variables.waste.reshape(variables.N, variables.N, variables.N)
    v_age = variables.vess_age.reshape(variables.N, variables.N, variables.N)
    act = variables.activity.reshape(variables.N, variables.N, variables.N)

    v_rad = v_age/(k_AR2+v_age)

    # print('Calculating Pressure')
    print(str(iteration))
    if (iteration )%20 == 0:
        variables.pres = np.zeros([1,variables.wlen])
        cindex = np.array(np.where(variables.celltype.flat != 0))
        vindex = np.array(np.where(variables.vess_tag.flat > 0))

        # Calculate CTP
        for tt in range(0, cindex.size):
            s=cindex[0,tt]
            
            if np.any(np.where(s+variables.index_bias > 8120601)):
               print('Error')

            # Density as defined by Equations (3c)
            density = np.sum(variables.celltype[0,s+variables.index_bias]>0)/len(variables.index_bias)
            
            # Euclidean distance in Equation (2), (4a)
            dist = sum((variables.nod3xyz[pres_bias,0:] - np.tile(variables.nod3xyz[s, 0:],(len(pres_bias),1)))**2) ** .5
            dist = np.array((np.sum((variables.nod3xyz[s + pres_bias, 0:] - np.tile(variables.nod3xyz[s, 0:],(len(pres_bias),1))) ** 2, axis=1))**.5)

            # Alpha in Equation (4b) is assumed to be constant
            amplitude=amplitude0

            # Sigma as defined by Equation (4c)
            sigma = (sigma0*density**2)/((density**2)+(0.5**2))+0.05

            # Pressure based on Gaussian function as in Equation (4a)
            variables.pres[0:,s+pres_bias] = variables.pres[0:,s+pres_bias] + amplitude * math.e ** ((-dist**2)/(2*sigma**2))
        #pres = Pres(cindex,vindex,sigma0,amplitude0,pres_bias)
        #pres.CTP()
        # Calculate VTP
        for ss in range(0,vindex.size):
            vind = vindex[0,ss]
            # Tumor cell density around vessel cell, here I assume vessel
            # only causes additional pressure inside tumor due to membrane
            # effect [Ref]

            density = np.sum(variables.celltype[0,vind+variables.index_bias]>0)/len(variables.index_bias)

            # Equation (4b), alpha0 set to be 0.01
            amplitude = (0.01*density**2)/(density**2 + 0.5**2)

            # Euclidean distance in Equation (4a)
            dist = np.array((np.sum((variables.nod3xyz[vind + pres_bias, 0:] - np.tile(variables.nod3xyz[vind, 0:],(len(pres_bias),1))) ** 2, axis=1))**.5)

            # Equation (4c)
            sigma = np.array((sigma0*density**2)/((density**2)+(0.5**2))+0.05)

             # Add vessel cell induced pressure
            variables.pres[0:,vind+pres_bias] = variables.pres[0:,vind+pres_bias] + amplitude * math.e ** ((-dist**2)/(2*sigma**2))

        #### Compute Pressure Gradient ####
        #pres.VTP()
        gradp = np.zeros((3,variables.wlen))

        variables.pres = variables.pres/p0

        # Using Central difference Approximation
        # Note: These three variables have different values to MatLAB due to roundoff issues 
        gradp[0,t1 - 1] = (variables.pres[0:,t1 + 1] - variables.pres[0:,t1 - 1])/(2*h)
        gradp[1, t1 - 1] = (variables.pres[0:, t1+variables.N] - variables.pres[0:, t1-variables.N])/(2*h) # Number of points between Y1 and Y2 is N
        gradp[2,t1 - 1] = (variables.pres[0:, t1+ variables.slen] - variables.pres[0:, t1-variables.slen])/(2*h)

        ux=-Kp*p0/(u0*L)*np.reshape(gradp[0,0:],(variables.N,variables.N,variables.N))
        uy=-Kp*p0/(u0*L)*np.reshape(gradp[1,0:],(variables.N,variables.N,variables.N))
        uz=-Kp*p0/(u0*L)*np.reshape(gradp[2,0:],(variables.N,variables.N,variables.N))

        ux=np.reshape(ux,(variables.N,variables.N,variables.N))
        uy=np.reshape(uy,(variables.N,variables.N,variables.N))
        uz=np.reshape(uz,(variables.N,variables.N,variables.N))

    p = np.reshape(variables.pres,(variables.N, variables.N, variables.N))
    weight = (cap_pres - p *p0)/ cap_pres
    weight[np.where(weight< 0)] = 0

    #### Nutrients Equation ####

    c1 = 1-6*Dn*tau*k/((L**2)*(h**2))
    c2=Dn*tau*k/((L**2)*(h**2))-tau*k*u0/L*ux[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h)
    c3=Dn*tau*k/((L**2)*(h**2))-tau*k*u0/L*uy[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    c4=Dn*tau*k/((L**2)*(h**2))-tau*k*u0/L*uz[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    c5=Dn*tau*k/((L**2)*(h**2))+tau*k*u0/L*ux[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    c6=Dn*tau*k/((L**2)*(h**2))+tau*k*u0/L*uy[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    c7=Dn*tau*k/((L**2)*(h**2))+tau*k*u0/L*uz[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h)

    rho_n = rho_n0*v_rad*weight 
    lambda_n = lambda_n0*act[1:variables.N-1,1:variables.N-1,1:variables.N-1]

    rho_nf = k*tau/n0*rho_n[1:variables.N-1,1:variables.N-1,1:variables.N-1]* v[1:variables.N-1,1:variables.N-1,1:variables.N-1]
    lambda_nf = k*tau/n0*lambda_n*c[1:variables.N-1,1:variables.N-1,1:variables.N-1]
    c_n = copy.deepcopy(n)
    n[1:variables.N-1,1:variables.N-1,1:variables.N-1] = ( c1*c_n[1:variables.N-1,1:variables.N-1,1:variables.N-1] 
    + c2*c_n[2:variables.N,1:variables.N-1,1:variables.N-1] 
    + c3*c_n[1:variables.N-1,2:variables.N,1:variables.N-1] 
    + c4*c_n[1:variables.N-1,1:variables.N-1,2:variables.N] 
    + c5*c_n[0:variables.N-2,1:variables.N-1,1:variables.N-1] 
    + c6*c_n[1:variables.N-1,0:variables.N-2,1:variables.N-1]
    + c7*c_n[1:variables.N-1,1:variables.N-1,0:variables.N-2]
    + rho_nf - lambda_nf )

    temp_n=np.array(n.flat)
    temp_n[boundary]=1
    # temp_n = np.reshape(temp_n,(variables.N, variables.N, variables.N))
    variables.nutr=copy.deepcopy(temp_n[0:].T)

    #### Metabolic Waste ####
    a1 = 1-6*Dw*tau*k/((L**2)*(h**2))
    a2=Dw*tau*k/((L**2)*(h**2))-tau*k*u0/L*ux[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h)
    a3=Dw*tau*k/((L**2)*(h**2))-tau*k*u0/L*uy[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    a4=Dw*tau*k/((L**2)*(h**2))-tau*k*u0/L*uz[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    a5=Dw*tau*k/((L**2)*(h**2))+tau*k*u0/L*ux[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    a6=Dw*tau*k/((L**2)*(h**2))+tau*k*u0/L*uy[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    a7=Dw*tau*k/((L**2)*(h**2))+tau*k*u0/L*uz[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h)

    w_new=w
    w_new[np.where(w_new < 1)]=1
    act= n/(n+1)*2*math.e**(-(w_new-1)**(4/.2))

    rho_w = rho_w0*act[1:variables.N-1,1:variables.N-1, 1:variables.N-1]
    lambda_w = lambda_w0*weight

    rho_wf = k*tau/w0*rho_w*c[1:variables.N-1,1:variables.N-1,1:variables.N-1]
    lambda_wf = k*tau/w0*lambda_w[1:variables.N-1,1:variables.N-1,1:variables.N-1]*v[1:variables.N-1,1:variables.N-1,1:variables.N-1]*v_rad[1:variables.N-1,1:variables.N-1,1:variables.N-1]
    
    w[1:variables.N-1,1:variables.N-1,1:variables.N-1]= (a1*w[1:variables.N-1,1:variables.N-1,1:variables.N-1] 
    + a2*w[2:variables.N,1:variables.N-1,1:variables.N-1] 
    + a3*w[1:variables.N-1,2:variables.N,1:variables.N-1]
    + a4*w[1:variables.N-1,1:variables.N-1,2:variables.N] 
    + a5*w[0:variables.N-2,1:variables.N-1,1:variables.N-1] 
    + a6*w[1:variables.N-1,0:variables.N-2,1:variables.N-1]
    + a7*w[1:variables.N-1,1:variables.N-1,0:variables.N-2]
    + rho_wf - lambda_wf)

    temp_w=np.array(w.flat)
    temp_w[boundary]=1
    # temp_n = np.reshape(temp_n,(variables.N, variables.N, variables.N)) 
    variables.waste=copy.deepcopy(temp_w[0:].T)

    #### TAF Equation ####
    s1 = 1-6*Dc*tau*k/((L**2)*(h**2))
    s2=Dc*tau*k/((L**2)*(h**2))-tau*k*u0/L*ux[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h)
    s3=Dc*tau*k/((L**2)*(h**2))-tau*k*u0/L*uy[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    s4=Dc*tau*k/((L**2)*(h**2))-tau*k*u0/L*uz[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    s5=Dc*tau*k/((L**2)*(h**2))+tau*k*u0/L*ux[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    s6=Dc*tau*k/((L**2)*(h**2))+tau*k*u0/L*uy[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h) 
    s7=Dc*tau*k/((L**2)*(h**2))+tau*k*u0/L*uz[1:variables.N-1,1:variables.N-1,1:variables.N-1]/(2*h)

    n_new=n[1:variables.N-1,1:variables.N-1,1:variables.N-1]
    n_new[np.where(n_new > 1)]=1

    rho_c = rho_c0*(1-n_new)
    lambda_c = lambda_c0*v_rad

    rho_cf = k*tau/c0*rho_c*c[1:variables.N-1,1:variables.N-1,1:variables.N-1]
    lambda_cf = k*tau/c0*lambda_c[1:variables.N-1,1:variables.N-1,1:variables.N-1]*v[1:variables.N-1,1:variables.N-1,1:variables.N-1]

    t[1:variables.N-1,1:variables.N-1,1:variables.N-1]= (s1*t[1:variables.N-1,1:variables.N-1,1:variables.N-1] 
    + s2*t[2:variables.N,1:variables.N-1,1:variables.N-1] 
    + s3*t[1:variables.N-1,2:variables.N,1:variables.N-1]
    + s4*t[1:variables.N-1,1:variables.N-1,2:variables.N] 
    + s5*t[0:variables.N-2,1:variables.N-1,1:variables.N-1] 
    + s6*t[1:variables.N-1,0:variables.N-2,1:variables.N-1]
    + s7*t[1:variables.N-1,1:variables.N-1,0:variables.N-2]
    + rho_cf - lambda_cf)

    t=np.array(t.flat)
    t[boundary]=0
    TAF = copy.deepcopy(t)
    # variables.activity = np.array(act.flat.T)
    variables.activity = copy.deepcopy(np.array(act.flat).T)

    ##### Tumor Model Growth ######
    cellindex = np.array(np.where(variables.celltype.flat > variables.nec))
    test_act = variables.activity[cellindex]
    test_cell_energy = variables.cell_energy[0,cellindex]
    variables.celltype[0,cellindex[np.where(test_act >= 0.5)]]=1                            # Active tumor cells
    variables.celltype[0,cellindex[np.where(test_act < 0.5)]]=0.95                          # Quescient tumor cells
    variables.celltype[0,cellindex[np.where(test_cell_energy <= 0.5)]]=variables.nec        # Necrosis Cells
    
    # Cell Proliferation
    div_index = np.where(variables.celltype.flat == .95)[0]
    #div_index = div_index[np.random.permutation(div_index.size)]                          # Randomizing the quiscent tumor cell order
    #np.random.shuffle(div_index)
    for ss in range(0, div_index.size):
        s = div_index[ss]
        variables.cell_energy[0,s] = variables.cell_energy[0,s] - 0.2

    div_index = np.where(variables.celltype.flat == 1)[0]
    # div_index = div_index[0,np.random.permutation(div_index.size)] 

    for ss in range(0,div_index.size):
        s = div_index[ss]
        prolif_energy = 30

        if np.any(variables.cell_energy[0,s] >= prolif_energy):
            cell.Celldivide_anisotropic_anatomy_v3(s)
        else:
            variables.cell_energy[0,s] = variables.cell_energy[0,s] + variables.activity[s] - 0.65*(variables.activity[s]/(variables.activity[s]+1))
    
    # 3D Angiogenesis
    startime = 500

    if iteration >= startime:

        variables.vessgrowth_flag = 1
        start= startime

        if iteration==startime:
            tip_index = vindexsave
        
        variables.vess_age[np.where(variables.vess_tag > 0)] = variables.vess_age[np.where(variables.vess_tag > 0)] + 1 # vess_age++ in each iteration
        tip_index =  np.union1d(np.where(variables.vess_tag==0.95)[0], variables.sprout_index) # Including the new sprouting points lying on preexisting blood vessel that were touched by floating hotpoints and other known tip cells.  
        #tip_index = tip_index[np.random.permutation(div_index.size)] # Randperm tip_index
        np.random.shuffle(tip_index)

        for i in range(0,len(tip_index)):
            s = tip_index[s]
            # Do Angiogenesis method
        magi = (0.3)*math.e(-3)
        spread.Spreadhotpoint(magi)


    if iteration > 200:
        if iteration < 375:
            if (iteration)%20 == 0:
                variables.days_forAndrea = round((iteration)/20)
                write.writeraw_forAndrea()
            else:
                if (iteration)%20 == 0:
                    variables.days_forAndrea = round((iteration)/20)
                    write.writeraw_forAndrea()
    




