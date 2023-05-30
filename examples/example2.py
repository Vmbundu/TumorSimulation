# Tumor Spiculation Example
spicules = []
mass = []
core = []
iter = 50
params = {
                "dirRandRange": [0.5, 1.5],
                "thickness": 4,
                "maxGrowth": 8,
                "numSpicules": 100,
            }
#testLesion is provided by you
with h5py.File(testLesion, "r") as f:
        mass = f["iter_{:d}".format(iter)][()]

massGen = Spicules(mass, params=params)
mass, spicules = massGen.generate(self.seed)

with h5py.File(SpiculFile, "a") as f:
                if 'iter_'+str(iter) in f:
                    del f['iter_'+str(iter)]
                f.create_dataset('iter_'+str(iter), data = mass.astype(np.uint8), compression="gzip", track_times=False)
