import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread("fullAna.jpg")
#print(img[648,1244])
plt.imshow(img)
plt.show()