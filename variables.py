import numpy as np

def init():
    global leng         # Virtual 3D computational domain length, unit: mm
    global N            # grid number
    global slen
    global wlen

    global nod3xyz      # wlen*3 Matrix: Matrix "Tumor Cell Location in 3D Spaces"

    global nutr         # 3D Matrix: Nutrients
    global pres         # 3D Matrix: Pressure
    global TAF          # 3D Matrix: TAF
    global waste
    global activity     # Cell activity
    global celltype     # 3D Matrix: Cell type
    global cell_energy  # 3D Matrix: "Tumor Cell energy for proliferation"

    global vess         # 3D Cell Matrix: "Endothelial Cell Information" 
    global vess_tag     # 3D Matrix denotes "Endothetial Cell" 1:normal EC, 0.95: Tip EC
    global vess_age     # 3D Matrix: "Vessel Cell Age"

    global hotpoint     # Vessel branching hotpoint

    global index_bias   # 3D Matrix: offsets toward neighboring grids from one certain grid point
    global stackvalue   # Vector: transfer parameter values for stack 
    global stackcount   # Stack operation counter
    global vessgrowth_flag
    global branchrecord # Record the vessel cells that have branched, I assure the vessel cells that have branched won't branch again
    global sprout_index
    global nec

    global video
    global counter1
    global counter2


    global spic
    global spic_index
    global local_data
    global days
    global days_forAndrea

    global ana_no
    global days
    global mass_dim

    # ================== Create Vectors ===================
    # The domain of simulations is a cube sized 1 cm^3
    # I assume the size of each tumor cell is 50 micrometers
    # Hence, in each cube length are 200 grids (or max 200 cells)
    leng = 10
    N = 201
    slen = N*N
    wlen = N*N*N


    # ================= Create Vectors ==================
    # Create vectors of intial values of all variables 
    # nutr = np.array([[1]*wlen]*1)
    nutr = np.ones((1,wlen))
    # waste = np.array([[1]*wlen]*1)
    waste = np.ones((1,wlen))
    # TAF = np.array([[0]*wlen]*1)
    TAF = np.zeros((1,wlen))

    # ================ Initialization other varibales ======================
    # pres = np.array([[0]*wlen]*1)
    pres = np.zeros((1,wlen), dtype=int)
    # activity = np.array([[0]*wlen]*1)  # Tumor cell activity should be zero at initial stage
    activity = np.zeros((1,wlen))
    # celltype = np.array([[0]*wlen]*1)
    celltype = np.zeros((1,wlen))
    # cell_energy = np.array([[0]*wlen]*1)
    cell_energy = np.zeros((1,wlen))
    vess = {}
    vess_tag = np.array([[0]*wlen]*1).astype(float)
    vess_age = np.array([[0]*wlen]*1)

    hotpoint = np.array([[0]*wlen]*1)
    branchrecord = np.array([[0]*wlen]*1)

    stackvalue = [0, 0]
    vessgrowth_flag = 0
    stackcount = 0

    xn = np.linspace(0,leng,N)
    [Y,X,Z] = np.meshgrid(xn,xn,xn)
    nod3xyz = np.array(np.vstack((Z.flatten(), Y.flatten(), X.flatten())).T) 

    nec = 1e-4

    counter1 = 0
    counter2 = 0

    spic = {
        "age": [[0]*wlen]*1,
        "dir": [[0]*wlen]*1,
        "par": [[0]*wlen]*1
            }
    spic_index = 0

    local_data = [[0]*wlen]*1

    index_bias = []

    sprout_index = []

    days_forAndrea = 0

    days = 0

    ana_no = 0
    
    mass_dim = 0



    
    
