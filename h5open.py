import h5py

with h5py.File('test.hdf5', 'r') as output:
        d1 = output['test_53'][()]

d1 = d1.astype('uint8')
fID = open('h5test.raw', 'w')
d1.tofile('h5test.raw')
fID.close