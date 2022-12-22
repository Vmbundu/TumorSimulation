global celltype 
global cell_energy
global N
global length
global slen
global nod3xyz
global activity
import numpy as np
import matplotlib.pyplot as plt
import variables
import math
def InitCancerCell():
    h = variables.leng/(variables.N-1)
    arr1 = np.array([[[2,17], [45, 78]], [[88, 92], [60, 76]],[[76,33],[20,18]]])
    num_init_cell = 5
    init_size = 0.25
    
    index0 = 110*variables.slen+100*variables.N+101
    init_cell = variables.nod3xyz[index0-1,0:]

    variables.celltype[0][index0-1] = 1
    variables.cell_energy[0][index0-1] = 15
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
            index = index0+x+y*variables.N+z*variables.slen
            init_cell = np.vstack((init_cell,variables.nod3xyz[index-1,0:]))
            variables.celltype[0][index-1] = 1
            variables.cell_energy[0][index-1] = 15
            variables.activity[0][index-1] = 1

    init_cell = init_cell*20
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(init_cell[0:,0],init_cell[0:,1], init_cell[0:,2])
    plt.xlim(0,200)
    plt.ylim(0,200)
    ax.set_zlim(0,200)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
