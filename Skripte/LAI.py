import numpy as np
import spectral.io.envi as envi
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def LAI(lai_infile):
    infile = lai_infile

    lib_in = envi.open(infile)
    spec = lib_in.spectra
    names = lib_in.names
    nspec = spec.shape[0]
    nbands = spec.shape[1]
    wavel = lib_in.bands.centers
    fwhm = lib_in.bands.bandwidths
    units = lib_in.bands.band_unit

    print(nbands)

    # now some plots:
    # spectra over wavelengths

    plt.plot(wavel, spec[0, :], color='brown')
    plt.plot(wavel, spec[1, :], color='red')
    plt.plot(wavel, spec[2, :], color='orange')
    plt.plot(wavel, spec[3, :], color='yellow')
    plt.plot(wavel, spec[4, :], color='lightgreen')
    plt.plot(wavel, spec[5, :], color='darkgreen')
    plt.ylabel('Reflektanz')
    plt.xlabel("Wellenlänge [Nanometer]")
    plt.legend(names)
    plt.show()

    # my_spec = spec[0, :]
    # plt.plot(my_spec)
    # plt.show()
    #
    # # now calculate the NDVI for it, suing NIR band 580 and red band 350:
    # print((my_spec[580] - my_spec[350]) / (my_spec[580] + my_spec[350]))

    # ... and now the NDVI for all spectra:
    ndvi = (spec[:, 580] - spec[:, 350]) / (spec[:, 580] + spec[:, 350])
    sr = (spec[:, 1250 - 350] / spec[:, 1050 - 350])
    diff_lai = (spec[:, (1725 - 350) - (970 - 350)])
    ind = ['NDVI', 'Simple Ratio', 'Difference LAI']


    lai = [0, 0.5, 1, 2, 3, 4]
    plt.plot(lai, ndvi, color='orange')
    plt.plot(lai, sr, color='red')
    plt.plot(lai, diff_lai, color='green')
    plt.xlabel('LAI')
    plt.ylabel('Value')
    plt.legend(ind)
    plt.show()

    print(ndvi, sr, diff_lai)

    # das Simple ration (SR) ist definiert als Ratio 1250 / 1050
    # da unsere Feldspektren das erste Band bei 350 nm haben, und dann in 1 nm Bandschritten gemessen wurden, ist das Ratio also
    # 1250 - 350 (da erstes band bei 350) nm   / 1050 - 350 (da erstes band bei 350) nm

    # for estimating how good our model is, we need to import some functions...

    ndvi = ndvi.reshape(-1, 1)
    sr = sr.reshape(-1, 1)
    diff_lai = diff_lai.reshape(-1, 1)
    band = spec[:, 450]
    band = band.reshape(-1, 1)

    # now calcualte the linear regression between NDVI and LAI:

    myfit_ndvi = LinearRegression(fit_intercept=True)
    myfit_ndvi.fit(ndvi, lai)
    print('Intercept : ', myfit_ndvi.intercept_)
    print('Gain coeff: ', myfit_ndvi.coef_)

    myfit_sr = LinearRegression(fit_intercept=True)
    myfit_sr.fit(sr, lai)
    print('Intercept : ', myfit_sr.intercept_)
    print('Gain coeff: ', myfit_sr.coef_)

    myfit_diff_lai = LinearRegression(fit_intercept=True)
    myfit_diff_lai.fit(diff_lai, lai)
    print('Intercept : ', myfit_diff_lai.intercept_)
    print('Gain coeff: ', myfit_diff_lai.coef_)

    # now the predicted LAI absed on the NDVI
    # as well as the model goodness (R2, RMSE, MSE)

    lai_linpred_ndiv = myfit_ndvi.predict(ndvi)
    print('R2 :', r2_score(lai, lai_linpred_ndiv))
    print('Mean Absolute Error:', mean_absolute_error(lai, lai_linpred_ndiv))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(lai, lai_linpred_ndiv)))
    print('Mean Squared Error:', mean_squared_error(lai, lai_linpred_ndiv))

    lai_linpred_sr = myfit_sr.predict(sr)
    print('R2 :', r2_score(lai, lai_linpred_sr))
    print('Mean Absolute Error:', mean_absolute_error(lai, lai_linpred_sr))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(lai, lai_linpred_sr)))
    print('Mean Squared Error:', mean_squared_error(lai, lai_linpred_sr))

    lai_linpred_diff_lai = myfit_diff_lai.predict(diff_lai)
    print('R2 :', r2_score(lai, lai_linpred_diff_lai))
    print('Mean Absolute Error:', mean_absolute_error(lai, lai_linpred_diff_lai))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(lai, lai_linpred_diff_lai)))
    print('Mean Squared Error:', mean_squared_error(lai, lai_linpred_diff_lai))

    # and plot the fit:

    plt.figure()
    plt.plot(lai, ndvi, 'ko', label="Original Data")
    plt.plot(lai_linpred_sr, sr, 'r-', label="Fitted Linear Model SR", color="red")
    plt.plot(lai_linpred_ndiv, ndvi, 'r-', label="Fitted Linear Model NDVI", color="orange")
    plt.plot(lai_linpred_diff_lai, diff_lai, 'r-', label="Fitted Linear Model DIFF LAI", color="green")
    plt.xlabel('LAI')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
