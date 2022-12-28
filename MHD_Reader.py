import re
import gzip
import numpy as np
import re
import copy
from string import Template

class MHD_Reader:
    """
        Object contructor for MHD_Reader 
    """

    def __init__(self):
        self.mhd = []

    def _read_mhd(self, filename):
            """
                Method that reads an mhd file 
                :param filename: Filename of the MHD file
                :returns: mhd info as an array
            """

            data = {}

            with open(filename, "r") as f:

                for line in f:

                    s = re.search(

                        "([a-zA-Z]*) = (.*)", line)

                    data[s[1]] = s[2]



                    if " " in data[s[1]]:

                        data[s[1]] = data[s[1]].split(' ')

                        for i in range(len(data[s[1]])):

                            if data[s[1]][i].replace(".", "").replace("-", "").isnumeric():

                                if "." in data[s[1]][i]:

                                    data[s[1]][i] = float(data[s[1]][i])

                                else:

                                    data[s[1]][i] = int(data[s[1]][i])

                    else:

                        if data[s[1]].replace(".", "").replace("-", "").isnumeric():

                            if "." in data[s[1]]:

                                data[s[1]] = float(data[s[1]])

                            else:

                                data[s[1]] = int(data[s[1]])

            #self.mhd = data
            return data

    def _load_phantom_array_from_gzip(self, mhd,results_folder, seed):
        """
            Method to convert full breast anatomy raw file into an array
            :param mhd: the mhd data array
            :returns: an array of the full breast anatomy
        """
        
        #self.mhd["ElementDataFile"] = self.elementData
        with gzip.open(mhd["ElementDataFile"], 'rb') as gz:  
            phantom = gz.read()

        with gzip.open('{:s}/pcl_{:d}.raw.gz'.format(results_folder, seed),'w') as m:
            m.write(phantom)
        
        return np.fromstring(phantom, dtype=np.uint8).reshape(

            mhd["DimSize"][2],

            mhd["DimSize"][1],

            mhd["DimSize"][0])

    def _read_loc(self, filename):
        
        result = []
        with open(filename) as gz:

            lines = gz.readlines()
        
        for line in lines:
            line = str(line)
            m = re.findall('\d+', line)
            if m[3] == '2':
                result.append(m)
        return result

    def raw_file_crop(self, array, x,y,z, size, results_folder, seed):
        range = int(size/2)
        arr = array[x - range:x+range+1, y - range:y+range+1, z - range: z+range+1]
        raw_pathway = "{:s}/pcl_{:d}_test.raw".format(results_folder, seed)
        fID = open(raw_pathway,'w')
        arr1 = np.array(arr, dtype="uint8")
        arr1.tofile(raw_pathway)
        fID.close()
        return arr

    def write_mhd(self, mhd, results_folder,seed):
        MHD_FILE = """ObjectType = $ObjectType

        NDims = $NDims

        BinaryData = $BinaryData

        BinaryDataByteOrderMSB = $BinaryDataByteOrderMSB

        CompressedData = $CompressedData

        TransformMatrix = $TransformMatrix

        Offset = $Offset

        CenterOfRotation = $CenterOfRotation

        ElementSpacing = $ElementSpacing

        DimSize = $DimSize

        AnatomicalOrientation = $AnatomicalOrientation

        ElementType = $ElementType

        ObjectType = $ObjectType

        ElementDataFile = $ElementDataFile"""
        with open("{:s}/pcl_{:d}.mhd".format(results_folder, seed), "w") as f:
                    src = Template(MHD_FILE)

                    template_arguments = copy.deepcopy(mhd)

                    template_arguments["ElementDataFile"] = "{:s}/pcl_{:d}.raw.gz".format(

                        results_folder, seed)

                    for key in template_arguments.keys():

                        if type(template_arguments[key]) is list:

                            template_arguments[key] = ' '.join(

                                map(str, template_arguments[key]))

                    result = src.substitute(template_arguments)

                    f.write(result)

 

        

#mdh_read = MHD_Reader('model_1/pc_3_crop.raw.gz')
#data = mdh_read._read_mhd('model_1/pc_3_crop.mhd')
#data['ElementDataFile'] = 'pc_4997_crop.raw.gz'
#arr = mdh_read._load_phantom_array_from_gzip()
#result = mdh_read._read_loc('model_1/pcl_3.loc')
#arr1 = mdh_read.raw_file_crop(arr,x,z,y)
#print('Done')
