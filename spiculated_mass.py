import numpy as np
import skimage
import progressbar
import h5py
import gzip


class Spicules():

    core = None

    params = {
        "dirRandRange": [0.5, 1.5],
        "thickness": 8,
        "maxGrowth": 10,
        "numSpicules": 100,
    }

    def __init__(self, core=None, params=dict(), cube_side=300, radius=30):

        self.origin = [0, cube_side // 2, cube_side // 2]
        
        self.tum = np.pad(core, 15, mode='constant')
        self.core = core.astype(bool)
        #self.core = np.pad(self.core, 15, mode='constant')
        
        if core is None:
            self.core, self.perimeter = self.create_circular_mask(
                cube_side, cube_side, cube_side, radius=radius)
        else:
            # get perimeter layer
            self.core = self.core.astype(bool)
            self.perimeter = self._get_perimeter_image(core)

        self.params.update(params)
        self.extend = 0
        self.count = 0

    def line(self, arr, x0, y0, z0, x1, y1, z1, thickness):
        """
            Bresenham's line algorithm

            :param arr: array of interest
            :param x0: x-coordinate of the beginning point
            :param y0: y-coordinate of the beginning point
            :param z0: z-coordinate of the beginning point
            :param x1: x-coordinate of the endpoint
            :param y1: y-coordinate of the endpoint 
            :param z1: z-coordinate of the endpoint 
            :param thickness: line thickness
            :return: array of the line

        """



        temp = np.zeros(arr.shape).astype(bool)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        x, y, z = x0, y0, z0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        sz = -1 if z0 > z1 else 1
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while (x0 != x1):
                x0 += sx
                if (p1 >= 0):
                    y0 += sy
                    p1 -= 2 * dx

                if (p2 >= 0):
                    z0 += sz
                    p2 -= 2 * dx

                p1 += 2 * dy
                p2 += 2 * dz
                temp[x0, y0, z0] = 1

        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while (y0 != y1):
                y0 += sy
                if (p1 >= 0):
                    x0 += sx
                    p1 -= 2 * dy

                if (p2 >= 0):
                    z0 += sz
                    p2 -= 2 * dy

                p1 += 2 * dx
                p2 += 2 * dz
                temp[x0, y0, z0] = 1
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while (z0 != z1):
                z0 += sz
                if (p1 >= 0):
                    y0 += sy
                    p1 -= 2 * dz

                if (p2 >= 0):
                    x0 += sx
                    p2 -= 2 * dz

                p1 += 2 * dy
                p2 += 2 * dx
                temp[x0, y0, z0] = 1

        # temp = scipy.ndimage.binary_dilation(
        #     temp, structure=skimage.morphology.ball(thickness // 2))

        # temp = scipy.signal.convolve(temp, skimage.morphology.ball(
        #     thickness // 2), mode="same") > 0
        temp = self.binary_dilation(
            temp, skimage.morphology.ball(thickness // 2))
        arr += temp

    def normalize(self, arr):
        """
            Normalizes array values

            :param arr: Array of interest 
            :returns: Nomralized array
        """
        dirSize = np.sqrt(arr[0]**2 + arr[1]**2 + arr[2]**2)
        return [arr[0] / dirSize, arr[1] / dirSize, arr[2] / dirSize]

    def create_circular_mask(self, h, w, d, center=None, radius=None):
        """
            Creates a circular image mask

            :param h: height
            :param w: width
            :param d: depth
            :param center: the location of the mask's center
            :param raidus: Radius of the circle
            :returns: the mask array 
        """
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2), int(d / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0],
                         h - center[1], d - center[1])

        Z, Y, X = np.ogrid[:d, :h, :w]
        dist_from_center = np.sqrt(
            (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)

        mask = dist_from_center <= radius
        perimeter = np.bitwise_and(
            dist_from_center > radius - 1, dist_from_center < radius + 1)
        return mask.astype(bool), perimeter.astype(bool)

    def saveHDF5(self, filename, volume):
        """
            Saves the spiculation file

            :param filename: The filename
            :param volume: the volume data of the tumor
            :returns: the tumor spiculation file 
        """
        # save in HDF
        with h5py.File(filename, "w") as hf:
            hf.create_dataset("volume", data=volume,
                              compression="gzip", track_times=False)

    def generate(self, seed):
        """
            Generates the tumor spicualtion

            :param seed: seed value
            :returns: tumor mass with spicualtion, spiculation data array
        """
        arr = np.zeros((self.core.shape[0], self.core.shape[1],
                       self.core.shape[2])).astype(bool)
        spicules = []
        np.random.seed(seed)
        bar = progressbar.ProgressBar(max_value=self.params["numSpicules"])
        for spicu in range(self.params["numSpicules"]):
            bar.update(spicu)
            flatPerimeter = np.where(self.perimeter.flatten())[0]
            posFlat = np.unravel_index(
                flatPerimeter[np.random.randint(0, len(flatPerimeter))], arr.shape)

            dir = [posFlat[0] - self.core.shape[0] // 2,
                   posFlat[1] - self.core.shape[1] // 2,
                   posFlat[2] - self.core.shape[2] // 2]
            dir = self.normalize(dir)

            activeNodes = [{
                "x": posFlat[0],
                "y": posFlat[1],
                "z": posFlat[2],
                "th": self.params["thickness"],
                "dir": dir,
                "active":True}]
            growth = 0

            while (growth < self.params["maxGrowth"]):

                growth += 1
                for idc in range(len(activeNodes)):
                    current = activeNodes[idc]
                    if not current["active"]:
                        continue
                    step = np.random.randint(5, 20)

                    dir = [dir[0] * (self.params["dirRandRange"][0] +
                                     (np.random.random() *
                                      (self.params["dirRandRange"][1] - self.params["dirRandRange"][0]))),
                           dir[1] * (self.params["dirRandRange"][0] +
                                     (np.random.random() *
                                      (self.params["dirRandRange"][1] - self.params["dirRandRange"][0]))),
                           dir[2] * (self.params["dirRandRange"][0] +
                                     (np.random.random() *
                                      (self.params["dirRandRange"][1] - self.params["dirRandRange"][0])))]

                    dir = self.normalize(dir)

                    posx = int(current["x"] + step * dir[0])
                    posy = int(current["y"] + step * dir[1])
                    posz = int(current["z"] + step * dir[2])

                    if posx < 0 or posy < 0 or posz < 0 or posx >= arr.shape[0] or posy >= arr.shape[1] or posz >= arr.shape[2]:
                        if self.count < 5:
                            self.count += 1
                            #activeNodes[idc]["active"] = False
                            continue
                        else:
                            self.core = np.uint8(self.core)
                            arr = np.uint8(arr)
                            self.core = np.pad(self.core, 50, mode='constant')
                            arr = np.pad(arr, 50, mode='constant')

                            arr = arr.astype(bool)
                            self.core = self.core.astype(bool)
                            
                            self.perimeter = self._get_perimeter_image(self.core)

                            for sp in spicules:
                                for act in sp:
                                    act["x"] = act["x"] + 50
                                    act["y"] = act["y"] + 50
                                    act["z"] = act["z"] + 50
                            #activeNodes[idc]["active"] = False
                            self.extend += 50
                            self.count = 0
                            continue
                        '''if self.extend:
                            self.core = np.uint8(self.core)
                            arr = np.uint8(arr)
                            self.core = np.pad(self.core, 50, mode='constant')
                            arr = np.pad(arr, 50, mode='constant')

                            arr = arr.astype(bool)
                            self.core = self.core.astype(bool)
                            
                            self.perimeter = self._get_perimeter_image(self.core)

                            for sp in spicules:
                                for act in sp:
                                    act["x"] = act["x"] + 50
                                    act["y"] = act["y"] + 50
                                    act["z"] = act["z"] + 50
                            #activeNodes[idc]["active"] = False
                            self.extend = False
                            continue
                        else:
                            activeNodes[idc]["active"] = False
                            continue '''

                    self.line(arr, current["x"], current["y"],
                              current["z"], posx, posy, posz, current["th"])

                    if np.random.rand() < 0.6:  # reduce thickness
                        current["th"] = current["th"] - 1
                        if current["th"] == 0:
                            current["active"] = False

                    activeNodes[idc] = {
                        "x": posx,
                        "y": posy,
                        "z": posz,
                        "dir": dir,
                        "th": current["th"],
                        "active": current["active"]}

                    if np.random.rand() < 0.10 and activeNodes[idc]["th"] > 3:
                        # new branch!
                        activeNodes.append({
                            "x": posx,
                            "y": posy,
                            "z": posz,
                            "dir": dir,
                            "th": activeNodes[idc]["th"],
                            "active": True})

            spicules.append(activeNodes)
        bar.finish()

        mass = np.zeros((self.core.shape[0], self.core.shape[1],
                        self.core.shape[2])).astype(bool)
        mass += self.core
        
        #for sp in spicules:
        #    mass += sp
        mass += arr
        return mass, spicules

        # self.saveHDF5("projectsML/Y2023/spiculations/examples3D/spiculated_{:04}.h5".format(
        #     seed), mass.astype(np.uint8))

    def generate_cont(self, seed, spicule, arr, core):
        """
            Continuous tumor spiculation growth  

            :param seed: seed value
            :param spicule: spiculation data array
            :param arr: arr of interests
            :param core: tumor model spicualtion grows on
            :returns: List of coordinates as voxels
        """
        self.core = core
        self.core = np.pad(self.core, self.extend, mode='constant')
        self.core = self.core.astype(bool)
        self.count = 0
        self.params["numSpicules"] = self.params["numSpicules"] + 20
        #self.params["maxGrowth"] = self.params["maxGrowth"] + 10
        #growth = 0
        np.random.seed(seed)
        bar = progressbar.ProgressBar(max_value=self.params["numSpicules"])
        spi_size = len(spicule)
        dir_inx = 0
        arr = self.binary_dilation(
            arr, skimage.morphology.ball(self.params["thickness"] // 2))
        
        for spic in range(self.params["numSpicules"]):
            bar.update(spic)
            # Add spicule thickening with binary_dilation

            if spic > spi_size - 1:
                flatPerimeter = np.where(self.perimeter.flatten())[0]
                posFlat = np.unravel_index(
                    flatPerimeter[np.random.randint(0, len(flatPerimeter))], arr.shape)

                dir = [posFlat[0] - self.core.shape[0] // 2,
                    posFlat[1] - self.core.shape[1] // 2,
                    posFlat[2] - self.core.shape[2] // 2]
                dir = self.normalize(dir)

                activeNodes = [{
                    "x": posFlat[0],
                    "y": posFlat[1],
                    "z": posFlat[2],
                    "th": self.params["thickness"],
                    "dir": dir,
                    "active":True}]
                spicule.append(activeNodes)
                
            else:
                activeNodes = spicule[spic]
                for act in spicule[spic]:
                        act["th"] = act["th"] + 1
                        act["active"] = True
                        dir_inx += 1

            growth = 0
            while (growth < self.params["maxGrowth"]):

                growth += 1
                for idc in range(len(activeNodes)):
                    current = activeNodes[idc]
                    if not current["active"]:
                        continue
                    step = np.random.randint(5, 20)
                    dir = current["dir"]

                    dir = [dir[0] * (self.params["dirRandRange"][0] +
                                     (np.random.random() *
                                      (self.params["dirRandRange"][1] - self.params["dirRandRange"][0]))),
                           dir[1] * (self.params["dirRandRange"][0] +
                                     (np.random.random() *
                                      (self.params["dirRandRange"][1] - self.params["dirRandRange"][0]))),
                           dir[2] * (self.params["dirRandRange"][0] +
                                     (np.random.random() *
                                      (self.params["dirRandRange"][1] - self.params["dirRandRange"][0])))]

                    dir = self.normalize(dir)

                    posx = int(current["x"] + step * dir[0])
                    posy = int(current["y"] + step * dir[1])
                    posz = int(current["z"] + step * dir[2])

                    if posx < 0 or posy < 0 or posz < 0 or posx >= arr.shape[0] or posy >= arr.shape[1] or posz >= arr.shape[2]:
                        if self.count < 5:
                            self.count += 1
                            #activeNodes[idc]["active"] = False
                            continue
                        else:
                            self.core = np.uint8(self.core)
                            arr = np.uint8(arr)
                            self.core = np.pad(self.core, 50, mode='constant')
                            arr = np.pad(arr, 50, mode='constant')

                            arr = arr.astype(bool)
                            self.core = self.core.astype(bool)
                            
                            self.perimeter = self._get_perimeter_image(self.core)

                            for sp in spicule:
                                for act in sp:
                                    act["x"] = act["x"] + 50
                                    act["y"] = act["y"] + 50
                                    act["z"] = act["z"] + 50
                            #activeNodes[idc]["active"] = False
                            self.extend += 50
                            self.count = 0
                            continue
                        
                        '''if self.extend == True:
                            self.core = np.uint8(self.core)
                            arr = np.uint8(arr)
                            self.core = np.pad(self.core, 50, mode='constant')
                            arr = np.pad(arr, 50, mode='constant')

                            arr = arr.astype(bool)
                            self.core = self.core.astype(bool)
                            
                            self.perimeter = self._get_perimeter_image(self.core)

                            for sp in spicule:
                                for act in sp:
                                    act["x"] = act["x"] + 50
                                    act["y"] = act["y"] + 50
                                    act["z"] = act["z"] + 50
                            #activeNodes[idc]["active"] = False
                            self.extend = False
                            continue
                        else:
                            activeNodes[idc]["active"] = False
                            continue '''

                    self.line(arr, current["x"], current["y"],
                              current["z"], posx, posy, posz, current["th"])

                    if np.random.rand() < 0.6:  # reduce thickness
                        current["th"] = current["th"] - 1
                        if current["th"] == 0:
                            current["active"] = False

                    activeNodes[idc] = {
                        "x": posx,
                        "y": posy,
                        "z": posz,
                        "dir": dir,
                        "th": current["th"],
                        "active": current["active"]}

                    if np.random.rand() < 0.10 and activeNodes[idc]["th"] > 3:
                        # new branch!
                        activeNodes.append({
                            "x": posx,
                            "y": posy,
                            "z": posz,
                            "dir": dir,
                            "th": activeNodes[idc]["th"],
                            "active": True})
            spicule[spicu] = activeNodes
                

        mass = np.zeros((self.core.shape[0], self.core.shape[1],
                        self.core.shape[2])).astype(bool)
        mass += self.core
        #for sp in spicules:
        #    mass += sp
        mass += arr
        #self.params["numSpicules"] = self.params["numSpicules"] + 10
        #self.params["maxGrowth"] = self.params["maxGrowth"] + 2
        return mass, spicule

    def binary_dilation(self, image, selem=None, out=None):
        '''Apply a binary-dilation operation to an image using a structuring element

        This function removes pixels to objects' perimeter in the image using
        a structuring element.

        Args:
            image (array-like):
                Image data as an array. If the input is not numpy.bool array,
                the data is converted to this type.
            selem (array-like):
                Structuring element as an boolean image of the same dimension of `image`
            out (numpy.bool array):
                Array to store the result of this operation. The length of the array
                must be the same as the input image.

        Returns:
            numpy.bool array: Dilated image (when `out` is `None`)
        '''
        dim = image.ndim
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        if not image.dtype == bool:
            image = image.astype(bool)
        if selem is None:
            if dim == 1:
                selem = np.ones(shape=[3], dtype=bool)
            elif dim == 2:
                selem = np.zeros(shape=[3, 3], dtype=bool)
                selem[1, :] = True
                selem[:, 1] = True
            elif dim == 3:
                selem = np.zeros(shape=[3, 3, 3], dtype=bool)
                selem[:, 1, 1] = True
                selem[1, :, 1] = True
                selem[1, 1, :] = True
        else:
            if not isinstance(selem, np.ndarray):
                selem = np.asarray(selem, dtype=bool)
            if not selem.dtype == bool:
                selem = selem.astype(bool)
            if any([num_pixels % 2 == 0 for num_pixels in selem.shape]):
                raise ValueError('Only structure element of odd dimension '
                                 'in each direction is supported.')
        perimeter_image = self._get_perimeter_image(image)
        perimeter_coords = np.where(perimeter_image)
        if out is None:
            return_out = True
            out = image.copy()
        else:
            return_out = False
            out[:] = image[:]

        if dim == 1:
            sx = selem.shape[0]
            rx = sx // 2
            lx = image.shape[0]
            for ix in perimeter_coords[0]:
                (jx_b, jx_e), (kx_b, kx_e) = self._generate_array_indices(
                    ix, rx, sx, lx)
                out[jx_b:jx_e] |= selem[kx_b:kx_e]

        if dim == 2:
            rx, ry = [n // 2 for n in selem.shape]
            lx = image.shape
            sx, sy = selem.shape
            lx, ly = image.shape
            for ix, iy in zip(perimeter_coords[0], perimeter_coords[1]):
                (jx_b, jx_e), (kx_b, kx_e) = self._generate_array_indices(
                    ix, rx, sx, lx)
                (jy_b, jy_e), (ky_b, ky_e) = self._generate_array_indices(
                    iy, ry, sy, ly)
                out[jx_b:jx_e, jy_b:jy_e] |= selem[kx_b:kx_e, ky_b:ky_e]

        if dim == 3:
            rx, ry, rz = [n // 2 for n in selem.shape]
            sx, sy, sz = selem.shape
            lx, ly, lz = image.shape
            for ix, iy, iz in zip(perimeter_coords[0], perimeter_coords[1], perimeter_coords[2]):
                (jx_b, jx_e), (kx_b, kx_e) = self._generate_array_indices(
                    ix, rx, sx, lx)
                (jy_b, jy_e), (ky_b, ky_e) = self._generate_array_indices(
                    iy, ry, sy, ly)
                (jz_b, jz_e), (kz_b, kz_e) = self._generate_array_indices(
                    iz, rz, sz, lz)
                out[jx_b:jx_e, jy_b:jy_e,
                    jz_b:jz_e] |= selem[kx_b:kx_e, ky_b:ky_e, kz_b:kz_e]

        if return_out:
            return out

    def binary_erosion(self, image, selem=None, out=None):
        '''Apply a binary-erosion operation to an image using a structuring element

        This function removes pixels around objects' perimeter in an image and returns
        the result as an image.
        See the `binary_dilation` function doc-string for the arguments and retuned value.

        :param image: image that is being eroded
        :param selem: structuring element 
        :param out: output array
        :return: the newly adjusted array
        '''
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        if not image.dtype == bool:
            image = image.astype(bool)

        out_image = self.binary_dilation(~image, selem, out)

        if out is None:
            return ~out_image
        else:
            out[:] = ~out[:]

    def binary_closing(self, image, selem=None, out=None):
        '''Apply a binary-closing operation to an image using a structuring element

        This function dilates an image and then erodes the dilation result.
        See the `binary_dilation` function doc-string for the arguments and retuned value.

        :param image: image that is being eroded
        :param selem: structuring element 
        :param out: output array
        :return: the newly adjusted array
        '''
        out_image = self.binary_erosion(
            self.binary_dilation(image, selem), selem, out)
        if out is None:
            return out_image

    def binary_opening(self, image, selem=None, out=None):
        '''Apply a binary-opening operation to an image using a structuring element

        This function erodes an image and then dilates the eroded result.
        See the `binary_dilation` function doc-string for arguments and retuned value.

        :param image: image that is being eroded
        :param selem: structuring element 
        :param out: output array
        :return: the newly adjusted array
        '''
        out_image = self.binary_dilation(
            self.binary_erosion(image, selem), selem, out)
        if out is None:
            return out_image

    def _get_perimeter_image(self, image):
        '''Return the image of the perimeter structure of the input image

        Args:
            image (Numpy array): Image data as an array

        Returns:
            Numpy array: Perimeter image
        '''
        dim = image.ndim
        if dim > 3:
            raise RuntimeError('Binary image in 4D or above is not supported.')
        count = np.zeros_like(image, dtype=np.uint8)
        inner = np.zeros_like(image, dtype=bool)

        count[1:] += image[:-1]
        count[:-1] += image[1:]

        if dim == 1:
            inner |= image == 2
            for i in [0, -1]:
                inner[i] |= count[i] == 1
            return image & (~inner)

        count[:, 1:] += image[:, :-1]
        count[:, :-1] += image[:, 1:]
        if dim == 2:
            inner |= count == 4
            for i in [0, -1]:
                inner[i] |= count[i] == 3
                inner[:, i] |= count[:, i] == 3
            for i in [0, -1]:
                for j in [0, -1]:
                    inner[i, j] |= count[i, j] == 2
            return image & (~inner)

        count[:, :, 1:] += image[:, :, :-1]
        count[:, :, :-1] += image[:, :, 1:]

        if dim == 3:
            inner |= count == 6
            for i in [0, -1]:
                inner[i] |= count[i] == 5
                inner[:, i] |= count[:, i] == 5
                inner[:, :, i] |= count[:, :, i] == 5
            for i in [0, -1]:
                for j in [0, -1]:
                    inner[i, j] |= count[i, j] == 4
                    inner[:, i, j] |= count[:, i, j] == 4
                    inner[:, i, j] |= count[:, i, j] == 4
                    inner[i, :, j] |= count[i, :, j] == 4
                    inner[i, :, j] |= count[i, :, j] == 4
            for i in [0, -1]:
                for j in [0, -1]:
                    for k in [0, -1]:
                        inner[i, j, k] |= count[i, j, k] == 3
            return image & (~inner)
        raise RuntimeError('This line should not be reached.')

    def _generate_array_indices(self, selem_center, selem_radius, selem_length, result_length):
        '''Return the correct indices for slicing considering near-edge regions

        Args:
            selem_center (int): The index of the structuring element's center
            selem_radius (int): The radius of the structuring element
            selem_length (int): The length of the structuring element
            result_length (int): The length of the operating image

        Returns:
            (int, int): The range begin and end indices for the operating image
            (int, int): The range begin and end indices for the structuring element image
        '''
        # First index for the result array
        result_begin = selem_center - selem_radius
        # Last index for the result array
        result_end = selem_center + selem_radius + 1
        # First index for the structuring element array
        selem_begin = -result_begin if result_begin < 0 else 0
        result_begin = max(0, result_begin)
        # Last index for the structuring element array
        selem_end = selem_length - (result_end - result_length) \
            if result_end > result_length else selem_length
        return (result_begin, result_end), (selem_begin, selem_end)


if __name__ == "__main__":
    seed = 2
    # massGen = Spicules()

    testLesion = "/projects01/VICTRE/miguel.lago/Y2023/spiculations/test.hdf5"

    with h5py.File(testLesion, "r") as f:
        mass = f["test_53"][()]
    massGen = Spicules(mass)

    mass = massGen.generate(seed)
    massGen.saveHDF5("./spiculated_{:04}.h5".format(
        seed), mass.astype(np.uint8))
