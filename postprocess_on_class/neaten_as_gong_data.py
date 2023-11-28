import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from astropy.io import fits

path = 'E:/Research/Program/SynopticMapPrediction/postprocess_on_class/'

for cr in range(2278,2279):
    # raw GONG fits: flipud to get synoptic maps
    raw_file = path + 'raw/' + 'mrzqs_c'+ str(cr) + '.fits'
    # raw_file = path + 'neaten/' + 'cr2259_neaten.fits'
    raw_fits = fits.open(raw_file)

    raw_Br = raw_fits[0].data
    raw_header = raw_fits[0].header
    # print(raw_header)

    plt.figure()
    plt.imshow(raw_Br,cmap='RdBu',vmin=-50,vmax=50)
    plt.colorbar()

    # predicted WSO mat: flipud to get synoptic maps
    pred_file = path + 'interp/' + 'cr' + str(cr) + '_interp.mat'
    pred_mat = scio.loadmat(pred_file)
    pred_Br = pred_mat['pred_Br_interp']

    plt.figure()
    plt.imshow(pred_Br,cmap='RdBu',vmin=-4,vmax=4)
    plt.colorbar()

    # replace original GONG data with predicted data
    neaten_fits = raw_fits
    neaten_fits[0].data = pred_Br

    plt.show()
    # plt.close()

    save_file = path + 'neaten/' + 'cr' + str(cr) + '_neaten.fits'
    # neaten_fits.writeto(save_file)