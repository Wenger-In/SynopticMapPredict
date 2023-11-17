import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from astropy.io import fits

path = 'E:/Research/Program/SynopticMapPrediction/postprocess_on_class/'

# example GONG fits: flipud to get synoptic maps
gong_file = path + 'mrzqs_c2048.fits'
gong_fits = fits.open(gong_file)

gong_Br = gong_fits[0].data
header = gong_fits[0].header
print(header)

plt.figure()
plt.imshow(gong_Br,cmap='RdBu',vmin=-50,vmax=50)
plt.colorbar()

# predicted WSO mat: flipud to get synoptic maps
pred_file = path + 'cr2259_interp.mat'
pred_mat = scio.loadmat(pred_file)
pred_Br = pred_mat['pred_Br_interp']

plt.figure()
plt.imshow(pred_Br,cmap='RdBu',vmin=-4,vmax=4)
plt.colorbar

# replace original GONG data with predicted data
gong_fits[0].data = pred_Br
new_Br = gong_fits[0].data

plt.figure()
plt.imshow(new_Br,cmap='RdBu',vmin=-4,vmax=4)
plt.colorbar()

plt.show()

gong_fits.writeto(path + 'neaten_cr2259.fits')