import numpy as np
# from osgeo import gdal
# from gdalconst import *
import scipy as scipy
from scipy import linalg, optimize

from scipy.optimize import curve_fit

from scipy import signal

# for the majority filter we need:
from scipy.ndimage.filters import generic_filter
from scipy.stats import mode

import sys
from sys import stdout

import time
import pandas as pd
import spectral.io.envi as envi

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.collections as collections

import seaborn as sns

from scipy.signal import savgol_filter

import os


def LAI():
    infile = "C:/GEO418/Daten_Messungen/Hausarbeit/AUSWERTUNG_LAI.slb.hdr"

    lib_in = envi.open(infile)
    spec = lib_in.spectra
    names = lib_in.names
    nspec = spec.shape[0]
    nbands = spec.shape[1]
    wavel = lib_in.bands.centers
    fwhm = lib_in.bands.bandwidths
    units = lib_in.bands.band_unit

    # now some plots:
    # spectra over wavelengths

    plt.plot(wavel, spec[0, :])
    plt.plot(wavel, spec[1, :])
    plt.plot(wavel, spec[2, :])
    plt.plot(wavel, spec[3, :])
    plt.plot(wavel, spec[4, :])
    plt.plot(wavel, spec[5, :])

    # plt.plot(wavel, spec[10,:])
    # plt.plot(wavel, spec[19,:])
    plt.ylabel('reflectance')
    plt.xlabel(units)
    # plt.show()

    my_spec = spec[0, :]
    # print(my_spec.shape)
    # plt.plot(my_spec)
    # plt.show()

    # now calculate the NDVI for it, suing NIR band 580 and red band 350:
    print((my_spec[580] - my_spec[350]) / (my_spec[580] + my_spec[350]))

    # ... and now the NDVI for all spectra:
    ndvi = (spec[:, 580] - spec[:, 350]) / (spec[:, 580] + spec[:, 350])

    # plt.plot(ndvi)
    # plt.show()

    lai = [0, 0.5, 1, 2, 3, 4]
    plt.plot(lai, ndvi)
    plt.xlabel('LAI')
    plt.ylabel('NDVI')
    # plt.show()

    # das Simple ration (SR) ist definiert als Ratio 1250 / 1050
    # da unsere Feldspektren das erste Band bei 350 nm haben, und dann in 1 nm Bandschritten gemessen wurden, ist das Ratio also
    # 1250 - 350 (da erstes band bei 350) nm   / 1050 - 350 (da erstes band bei 350) nm

    sr = (spec[:, 1250 - 350] / spec[:, 1050 - 350])
    plt.plot(lai, sr)
    plt.xlabel('LAI')
    plt.ylabel('Simple Ratio')
    # plt.show()

    # for estimating how good our model is, we need to import some functions...

    print("NDVI shape vorher:", ndvi.shape)
    ndvi = ndvi.reshape(-1, 1)
    print("Nachher: ", ndvi.shape)

    print("sr shape vorher:", sr.shape)
    sr = sr.reshape(-1, 1)
    print("Nachher: ", sr.shape)

    band = spec[:, 450]
    print("band shape vorher:", band.shape)
    band = band.reshape(-1, 1)
    print("Nachher: ", band.shape)

    # now calcualte the linear regression between NDVI and LAI:

    myfit = LinearRegression(fit_intercept=True)
    myfit.fit(ndvi, lai)
    print('Intercept : ', myfit.intercept_)
    print('Gain coeff: ', myfit.coef_)

    # now the predicted LAI absed on the NDVI
    # as well as the model goodness (R2, RMSE, MSE)

    lai_linpred = myfit.predict(ndvi)
    print('R2 :', r2_score(lai, lai_linpred))
    print('Mean Absolute Error:', mean_absolute_error(lai, lai_linpred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(lai, lai_linpred)))
    print('Mean Squared Error:', mean_squared_error(lai, lai_linpred))

    # and plot the fit:

    plt.figure()
    plt.plot(lai, ndvi, 'ko', label="Original Data")
    plt.plot(lai_linpred, ndvi, 'r-', label="Fitted Linear Model")
    plt.xlabel('LAI')
    plt.ylabel('NDVI')
    plt.legend()
    plt.show()