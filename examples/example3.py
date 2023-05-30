#Victre Pipeline example
import Constants
import Pipeline

seed = 1

# Phantom file provided by you 
pline = Pipeline(seed=seed,
                     lesion_file=lesion_file,
                     results_folder="./results",
                     #phantom_file="./Victre/results/{:d}/pc_{:d}_crop.raw.gz".format(seed,seed),
                     arguments_generation=Constants.VICTRE_DENSE,
                     roi_sizes=roi_sizes
                     )

pline.generate_phantom()

pline.compress_phantom()

pline.crop()

pline.grow_lesion(time_array=[50], zoom=1.0)

#testLesion and SpicuFiles are provided by you
pline.spiculation(testLesion, 
                SpiculFile, iter=[302,303,304,305,306,307])

#lesion_file provided by you
pline.insert_lesions(lesion_file, lesion_type=Constants.VICTRE_SPICULATED, n=1, locations = [[667,667,2000]])
