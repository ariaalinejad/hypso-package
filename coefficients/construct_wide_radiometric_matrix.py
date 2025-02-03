import pandas as pd
import numpy as np
import glob
import os 

directory = os.path.dirname(os.path.realpath(__file__))

os.chdir(directory)

files = glob.glob("*_full.npz", root_dir=directory)

print(directory)
print(files)


for file in files:
    # TODO double check that these values actually are correct for radiometric calibration preformed
    # standard aoi values 
    bin_factor = 9
    aoi_x = 428
    aoi_y = 62
    row_count = 1092
    column_count = 1080

    # calculate start and stop values
    x_start = aoi_x
    x_stop  = aoi_x + column_count
    y_start = aoi_y
    y_stop  = aoi_y + row_count

    # load the full matrix, construct wide matrix based on aoi and binning factor
    full_matrix = np.load(file)
    wide_matrix = full_matrix['arr_0'][y_start:y_stop, x_start: x_stop].reshape(row_count, -1, bin_factor).mean(axis=2).reshape(row_count, -1)

    # save new matrix in same directory
    np.savez(str(os.path.splitext(file)[0])[:-5] + "_wide.npz", wide_matrix)