import cv2
from PIL import Image
import numpy as np
from bmp_reader import BMPReader
import math
import Vas3_functions as v3
# Generate a 3D vasculature image based on hand draw 2D
# vasculature image named vesselimage2D.bmp
data = np.array(BMPReader('vesselimage2D.bmp').get_pixels())
#data = cv2.normalize(data.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
data = (data - np.min(data))/(np.max(data) - np.min(data))
image01 = np.array(data[0:,0:,0])
N = np.size(image01)
N = N ** (1/2)
test = np.array(image01.flat)
index = np.nonzero(test<1)
index = np.array(index).transpose()
np.savetxt('index.txt', index, fmt='%d')
ind = np.linspace(1,1000,1000)

# Generate a 3D curve
[y,x] = np.meshgrid(ind,ind)

z1 = list(map(v3.z1_fuc,x))
z1 = np.array(z1)
z2 = list(map(v3.z2_fuc,x,y))
z2 = np.array(z2)
z3 = list(map(v3.z3_fuc,x))
z3 = np.array(z3)
z4 = list(map(v3.z4_fuc,y))
z4 = np.array(z4)
z5 = list(map(v3.z5_fuc,x,y))
z5 = np.array(z5)
total = z1+z2+z3+z4+z5 + ([[50]*1000]*1000)
z = np.array(total)/2

#Draw 3D images after Processing
#yy0 = (np.array(index) - (np.array(index)) % N)/N
yy0 = index - (index % N)/N
#xx0 = np.array(index) - np.array(yy0)*N
xx0 = index - (yy0*N)
zz0 = test[index] + z.flatten()[index]

#yy = ((np.array(index) - (np.array(index)) % N)/N)/5
yy = index - ((index % N)/N)/5
#xx = (np.array(index) - np.array(yy0)*N)/5
xx = index - (yy0*N)/5
zz = (test[index] + z.flatten()[index] - 42)/147*200

index = np.where(xx==0)
index = np.array(index)
del xx[index]
#xx[index] = []
#yy[index] = []
#zz[index] = []
