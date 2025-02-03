import numpy as np
import os 

directory = os.path.dirname(os.path.realpath(__file__))

os.chdir(directory)

file = "h2_spectral_calibration_wavelengths_center_row.npz"

print(directory)
print(file)

# TODO double check that these values actually are correct for the calibration that was performed

# standard aoi values 
bin_factor = 9
aoi_x = 428
aoi_y = 62
column_count = 1080

# calculate start and stop values
x_start = aoi_x
x_stop  = aoi_x + column_count

# load the full matrix, construct wide matrix based on aoi and binning factor
full_vector = np.load(file)
binned_vector = full_vector['arr_0'][x_start: x_stop].reshape( -1, bin_factor).mean(axis=1).reshape( -1)

# save new matrix in same directory
np.savez("spectral_bands_HYPSO-2.npz", binned_vector)