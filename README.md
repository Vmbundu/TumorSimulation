# TumorSimulation
The code here allows one to grow tumor in silico with and without spiculation. This feature will be added to the FDA VICTRE Pipeline. This repository is here to showcase these added features and the code behind it.

Usage
-----
In order to properly use this code, it is recommended that you have the access to the full VICTRE Pipeline. However, here are [examples](https://github.com/Vmbundu/TumorSimulation/tree/main/examples) that you can test and try out.

| File Name  | Description |
| ------------- | ------------- |
| `p_1.raw.gz`  | original phantom  |
| `pc_1.raw.gz`  | compressed phantom  |
| `pc_1_crop.raw.gz` | cropped compressed phantom |
| `pcl_1.raw.gz` | compressed original phantom with the inserted lesions|
| `pcl_1_resTumor(1).hdf5` | lesion at different timepoints|
| `pcl_1_res(1).hdf5` | lesion at different timepoints inserted in the ROI|
| `pcl_1_resFull(1).hdf5` | lesion at different timepoints inserted in the full phantom|
| `pcl_1.loc` | file containing the coordinates of the inserted lesions in the phantom (last number is the lesion type: `1` for calcification clusters, `2` for masses)|
| `spiculation_001.h5` | file that contains tumors with spiculation added|
| `projection_DM1.raw` | contains the DM projection  in raw format |
| `reconstruction1.raw` | contains the DBT reconstruction in raw format |
| `ROIs.h5` | contains the lesion-present and lesion-absent regions of interest|
| `ROIs` | subfolder will also contain the ROIs in raw format (size is specified in the code, `109 x 109 x 9` in the examples, `T = 1` for calcification clusters, `T = 2` for masses) |
| `ROIs\ROI_DM_XX_typeT` | DM cropped image for lesion number `XX` of lesion type `T` (absent regions will have `T < 0`) |
| `ROIs\ROI_DBT_XX_typeT`| DBT cropped volume for lesion number `XX` of lesion type `T` (absent regions will have `T < 0`)




File list
---------

The organization of the Victre pipeline python class is as follows:
| File Name | Description |
| --------- | ----------- |
| `Pipeline.py` | Main python class including all code necessary to run the Victre pipeline |
| `TumorSim.py*` | python class including all code necessary to run tumor simulations |
| `writetoraw_forAndrea.py*` | python class that writes tumor generation results to files |
| `spiculation_mass.py*` | python class that adds spicualtion growth to tumors |
| `Constants.py` | Helper file that includes default parameters for all the steps of the pipeline |
| `scaleImage.py*` | Python class that contains functions provide image resizing |
| `DetectedBoundary.py*` | Python class that dectects the boundary of the tumor ROI |
| `Exceptions.py` | Helper file that defines Victre exceptions |
| `breastMass` | Folder including the [breastMass](https://github.com/DIDSR/breastMass) software, needs to be pre-compiled |
| `compression` | Folder including the [breastCompress](https://github.com/DIDSR/breastCompress) software, needs to be pre-compiled |
| `generation` | Folder including the [breastPhantom](https://github.com/DIDSR/breastPhantom) software, needs to be pre-compiled |
| `projection` | Folder including the [MC-GPU](https://github.com/DIDSR/VICTRE_MCGPU) software, needs to be pre-compiled |
| `reconstruction` | Folder including the [FBP](https://github.com/DIDSR/VICTRE/tree/master/FBP%20DBT%20reconstruction%20in%20C) software, needs to be pre-compiled |
| `ModelObserver` | Folder including the Model Observer class |

"*" - Denotes the files I added
