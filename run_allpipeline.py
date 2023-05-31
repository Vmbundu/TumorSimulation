from Victre.Pipeline import Pipeline
from Victre import Constants
import os
import numpy as np
import argparse
import h5py

# Setting variabes to the arguments passed
parser = argparse.ArgumentParser(
    description='Runs on VICTRE pipeline task for DM vs DBT.')
parser.add_argument('--step', type=int, default=1,
                    help='Step 1: lesion insertion. Step 2: projection. Step 3: reconstruction and ROI saving.')
parser.add_argument('--density', type=str, default="dense")
parser.add_argument('--results', type=str, help='Results folder.',
                    default="/home/vanday.bundu/vv/Victre/results")
parser.add_argument('--size', type=float, help='Lesion size.', default=5.0)

parser.add_argument('--signal-only', dest="only", type=int, default=0,
                    help='Activate signal only.')

parser.add_argument('--spiculated', type=int, default=13,
                    help='Seed for spiculated mass.')

#Initialization
arguments = vars(parser.parse_args())

results_folder = arguments["results"]
seed = os.getenv('SGE_TASK_ID')

if seed is None:
    seed = 1

seed = int(seed)

TASK_ID = os.getenv('SGE_TASK_ID')

if TASK_ID is None:
    TASK_ID = 1

'''lesion_file = "./lesions/spiculated/mass_{:d}_size{:.2f}.h5".format(
    arguments["spiculated"], arguments["size"])'''
#lesion_file = "./Victre/results/{:d}/pcl_2_res(2).hdf5".format(seed)

# lesion_file = "./Victre/results/{:d}/0.h5".format(seed)
lesion_file = "./Victre/results/{:d}/test_tumor.hdf5".format(seed)

roi_sizes = {Constants.VICTRE_SPICULATED: [109, 109, 9],
             Constants.VICTRE_CLUSTERCALC: [65, 65, 5]}

histories = {"dense": 7.8e9,
             "hetero": 1.02e10,
             "scattered": 2.04e10,
             "fatty": 2.22e10}

arguments_mcgpu = {"number_histories": histories[arguments["density"]]}

if arguments["density"] == "fatty":
    arg_gen = Constants.VICTRE_FATTY
elif arguments["density"] == "scattered":
    arg_gen = Constants.VICTRE_SCATTERED
elif arguments["density"] == "hetero":
    arg_gen = Constants.VICTRE_HETERO
else:
    arg_gen = Constants.VICTRE_DENSE

density = arguments["density"]

spectrum_file = "./Victre/projection/spectrum/W28kVp_Rh50um_Be1mm.spc"

# Phantom and Lesion Generation
if arguments["step"] == 1:
    pline = Pipeline(seed=seed,
                     lesion_file=lesion_file,
                     results_folder=arguments["results"],
                     #phantom_file="./Victre/results/{:d}/pc_{:d}_crop.raw.gz".format(seed,seed),
                     arguments_generation=arg_gen,
                     roi_sizes=roi_sizes
                     )

    #pline.generate_phantom()

    #pline.compress_phantom()

    #pline.crop()

    #pline.compute_location()

    pline.grow_lesion(time_array=[50], zoom=1.0)

    # Picks the timoepoint and location of the breast phantom used in the pcl.raw file 
    #pline.extract_timepoint(timepoint=16,location=1)

    #Add Spiculation
    #pline.spiculation(testLesion="./Victre/results/{:d}/pcl_{:d}_resTumor(1).hdf5".format(seed,seed), 
    #            SpiculFile="./Victre/results/{:d}/spiculated_{:03}.h5".format(seed,seed), day=[12,13,14,15,16,17])

    #pline.insert_lesions(lesion_file = "./Victre/results/{:d}/pcl_{:d}_resTumor(1).hdf5".format(seed, seed), lesion_type=Constants.VICTRE_SPICULATED, n=1, locations = [[667,667,2000]])
    pline.insert_lesions(lesion_file = "./Victre/results/{:d}/spiculated_{:03}.h5".format(seed, seed), lesion_type=Constants.VICTRE_SPICULATED, n=1, locations = [[667,667,2000]])
    # odd numbers will not have a lesion inserted!
  #  if int(TASK_ID) % 2 != 0:
  #      pline.insert_lesions(lesion_type=Constants.VICTRE_SPICULATED,
  #                           n=4)
    # Adds lesion absent regions of interest
    #pline.add_absent_ROIs(lesion_type=Constants.VICTRE_SPICULATED,
    #                       n=4)

# Projection
elif arguments["step"] == 2:
    if "fatty" in arguments["density"]:
        spectrum_file = "./Victre/projection/spectrum/W30kVp_Rh50um_Be1mm.spc"
        arguments_mcgpu.update(
            {"fam_beam_aperture": [17.0, 13.2]})

    # ELENA: if you want to do only DM (not DBT) use these lines
    # arguments_mcgpu.update(
    #     {"number_projections": 1,
    #      "number_histories": arguments_mcgpu["number_histories"] * 25 * 2 / 3}
    # )
    pline = Pipeline(# seed=int(TASK_ID),
                     seed=seed,
                     results_folder=results_folder,
                     # ELENA: if you want to get a pregenerated phantom, I have some on this folder,
                     # use it by uncommenting the next line
                     # phantom_file=f"/projects01/VICTRE/miguel.lago/phantoms/{density}/{TASK_ID}/pc_{TASK_ID}_crop.raw.gz",
                     #flatfield_DBT='./Victre/results/2/scattered.raw',
                     #latfield_DM='./Victre/results/2/scatteredDBT.raw',
                     #phantom_file="{:s}/{:d}/pc_{:d}_crop.raw.gz".format(arguments["results"], seed, seed),
                     spectrum_file=spectrum_file,
                     lesion_file=lesion_file,
                     arguments_mcgpu=arguments_mcgpu) 
    
    pline.project()

# Reconstruction
elif arguments["step"] == 3:
    pline = Pipeline(#seed=int(TASK_ID),
                     seed=seed,
                     results_folder=results_folder,
                     lesion_file=lesion_file)
    pline.reconstruct()

    # Calculates the segmentation of the DBT model using the phantom
    mask = pline.get_DBT_segmentation()
    with h5py.File("{:s}/{:d}/reconstruction_mask{:d}.h5".format(results_folder, int(TASK_ID), int(TASK_ID)), "w") as hf:
        hf.create_dataset("dbt", data=mask, compression="gzip")

    # Saves the ROI as well as the DM/DBT images as DICOMs
    pline.save_ROIs(roi_sizes=roi_sizes)
    pline.save_DICOM("dm")
    pline.save_DICOM("dbt")
else:
    print("ERROR: Step number not valid.")
