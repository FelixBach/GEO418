import spectral.io.envi as envi
import matplotlib.pyplot as plt


def spectral():
    infile = "C:/GEO418/Daten_Messungen/Hausarbeit/AUSWERTUNG_Arten.slb.hdr"

    lib_in = envi.open(infile)
    spec = lib_in.spectra
    names = lib_in.names
    nspec = spec.shape[0]
    nbands = spec.shape[1]
    wavel = lib_in.bands.centers
    fwhm = lib_in.bands.bandwidths
    units = lib_in.bands.band_unit

    print("Namen der Messungen:", names)
    print("Spektren:", nspec)
    print("Bänder:", nbands)

    print(spec.shape)
    for i in range(nspec):
        plt.plot(wavel, spec[i])
    # plt.plot(wavel, spec[0, :])
    # plt.plot(wavel, spec[4, :])
    # plt.plot(wavel, spec[10, :])
    # #
    plt.ylabel('reflectance')
    plt.xlabel(units)
    plt.show()
