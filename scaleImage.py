from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py


def resize(im, new_w, new_h, keepStructure=False):
    width, height = im.shape

    newIm = np.zeros((new_w, new_h))

    new_wP = int(width * width / new_w)
    new_hP = int(height * height / new_h)

    step_w, step_h = width / new_w / 2, height / new_h / 2

    for new_y in range(new_h):
        # old_y = int(round(new_y * (new_h - 1) / (height - 1)))
        old_y = ((new_y * (new_hP - 1) / (height - 1)))
        if old_y < 0:
            old_y = 0
        if old_y >= height:
            old_y = height - 1
        for new_x in range(new_w):
            # old_x = int(round(new_x * (new_w - 1) / (width - 1)))
            old_x = ((new_x * (new_wP - 1) / (width - 1)))
            if old_x < 0:
                old_x = 0
            if old_x >= width:
                old_x = width - 1
            if keepStructure:
                around = im[np.floor(np.max((0, old_x - step_w))).astype(int):np.ceil(old_x + step_w).astype(
                    int) + 1, np.floor(np.max((0, old_y - step_h))).astype(int):np.ceil(old_y + step_h).astype(int) + 1]
                loc = np.argmax(around)
                loc = np.unravel_index(loc, around.shape)
                newIm[new_x, new_y] = im[np.floor(np.max((0, old_x - step_w))).astype(
                    int) + loc[0], np.floor(np.max((0, old_y - step_h))).astype(int) + loc[1]]

            else:
                loc = (int(round(old_y)), int(round(old_x)))
                newIm[new_y, new_x] = im[loc[0], loc[1]]

    return newIm


def resize3D(im, new_w, new_h, new_d, keepStructure=False):
    depth, width, height = im.shape

    newIm = np.zeros((new_d, new_w, new_h))

    step_d, step_w, step_h = depth / new_d / 2, width / new_w / 2, height / new_h / 2

    new_wP = int(width * width / new_w)
    new_hP = int(height * height / new_h)
    new_dP = int(depth * depth / new_d)

    for new_z in range(new_d):
        # old_y = int(round(new_y * (new_h - 1) / (height - 1)))
        old_z = ((new_z * (new_dP - 1) / (depth - 1)))
        if old_z < 0:
            old_z = 0
        if old_z >= depth:
            old_z = depth - 1
        for new_y in range(new_h):
            # old_y = int(round(new_y * (new_h - 1) / (height - 1)))
            old_y = ((new_y * (new_hP - 1) / (height - 1)))
            if old_y < 0:
                old_y = 0
            if old_y >= height:
                old_y = height - 1
            for new_x in range(new_w):
                # old_x = int(round(new_x * (new_w - 1) / (width - 1)))
                old_x = ((new_x * (new_wP - 1) / (width - 1)))
                if old_x < 0:
                    old_x = 0
                if old_x >= width:
                    old_x = width - 1
                if keepStructure:
                    around = im[np.floor(np.max((0, old_z - step_d))).astype(int):np.ceil(old_z).astype(int) + 1,
                                np.floor(np.max((0, old_x - step_w))).astype(int):np.ceil(old_x).astype(int) + 1,
                                np.floor(np.max((0, old_y - step_h))).astype(int):np.ceil(old_y).astype(int) + 1]
                    loc = np.argmax(around)
                    loc = np.unravel_index(loc, around.shape)
                    newIm[new_z, new_x, new_y] = im[np.floor(np.max((0, old_z - step_d))).astype(int) + loc[0],
                                                    np.floor(np.max((0, old_x - step_w))).astype(
                                                        int) + loc[1],
                                                    np.floor(np.max((0, old_y - step_h))).astype(int) + loc[2]]

                else:
                    loc = (int(round(old_z)), int(
                        round(old_y)), int(round(old_x)))
                    newIm[new_z, new_y, new_x] = im[loc[0], loc[1], loc[2]]

    return newIm


with h5py.File("MID/pcl_4_res.hdf5", "r") as hf:
    im3D = hf["day_1"][()]

downSize = int(im3D.shape[1] * 0.1)

im = im3D[im3D.shape[0] // 2, :, :]
# im = np.array(Image.open("MID/ligaments.bmp"))
# im.show()

reIm = resize(im, downSize, downSize, keepStructure=False)
reIm2 = resize(im, downSize, downSize, keepStructure=True)

f1 = plt.figure(1, (16, 6))

ax1 = f1.add_subplot(1, 3, 1)
ax1.imshow(im)
ax1.set_title("ORIGINAL RESOLUTION")

ax2 = f1.add_subplot(1, 3, 2)
ax2.imshow(reIm)
ax2.set_title("RESCALED (NN)")

ax3 = f1.add_subplot(1, 3, 3)
ax3.imshow(reIm2)
ax3.set_title("RESCALED (MAX)")


# 3D

im3D = im3D[im3D.shape[0] // 2 - 5:im3D.shape[0] // 2 + 5, :, :]
downSize3D = int(im3D.shape[0] * 0.7)

reIm3D = resize3D(im3D, downSize, downSize, downSize3D, keepStructure=False)
reIm3D2 = resize3D(im3D, downSize, downSize, downSize3D, keepStructure=True)

f2 = plt.figure(2, (16, 16))

ax1 = f2.add_subplot(1, 3, 1)
ax1.imshow(np.vstack(im3D))
ax1.set_title("ORIGINAL RESOLUTION 3D")

ax2 = f2.add_subplot(1, 3, 2)
ax2.imshow(np.vstack(reIm3D))
ax2.set_title("RESCALED 3D (NN)")

ax3 = f2.add_subplot(1, 3, 3)
ax3.imshow(np.vstack(reIm3D2))
ax3.set_title("RESCALED 3D (MAX)")

plt.show()
