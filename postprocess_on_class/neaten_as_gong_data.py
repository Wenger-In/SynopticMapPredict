import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from astropy.io import fits

path = 'E:/Research/Program/SynopticMapPrediction/postprocess_on_class/'

# raw GONG fits: flipud to get synoptic maps
raw_file = path + 'raw/' + 'mrzqs_c2048.fits'
# gong_file = path + 'neaten/' + 'neaten_cr2259.fits'
raw_fits = fits.open(raw_file)

raw_Br = raw_fits[0].data
raw_header = raw_fits[0].header
print(raw_header)

plt.figure()
plt.imshow(raw_Br,cmap='RdBu',vmin=-50,vmax=50)
plt.colorbar()

# predicted WSO mat: flipud to get synoptic maps
pred_file = path + 'cr2259_interp.mat'
pred_mat = scio.loadmat(pred_file)
pred_Br = pred_mat['pred_Br_interp']

plt.figure()
plt.imshow(pred_Br,cmap='RdBu',vmin=-4,vmax=4)
plt.colorbar()

# replace original GONG data with predicted data
gong_fits[0].data = pred_Br
new_Br = gong_fits[0].data

plt.figure()
plt.imshow(new_Br,cmap='RdBu',vmin=-4,vmax=4)
plt.colorbar()

plt.show()

# gong_fits.writeto(path + 'neaten_cr2259.fits')