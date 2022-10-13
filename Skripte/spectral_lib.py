import spectral.io.envi as envi
import matplotlib.pyplot as plt


def spectral(spectral_infile):
    infile = spectral_infile

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
    names_leg = names[0], names[4], names[8]
    print(names_leg)

    print(spec.shape)
    for i in range(nspec):
        plt.plot(wavel, spec[i])
    plt.ylabel('reflectance')
    plt.xlabel(units)
    plt.show()
    # plt.savefig("C:/GEO418/Daten_Messungen/Hausarbeit/AUSWERTUNG_Arten_all_spec.png", dpi='figure', format=None,
    #             metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)

    plt.plot(wavel, spec[0, :])
    plt.plot(wavel, spec[4, :])
    plt.plot(wavel, spec[10, :])
    plt.ylabel('Reflektanz')
    plt.xlabel('Wellenlänge [Mikrometer]')
    plt.legend(names_leg)
    plt.show()
    # plt.savefig("C:/GEO418/Daten_Messungen/Hausarbeit/AUSWERTUNG_Arten_arten_spec.png", dpi='figure', format=None,
    #             metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)
