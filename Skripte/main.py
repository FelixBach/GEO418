import spectral_lib
import LAI
import SM


def main():
    spectral_infile = "C:/GEO418/Daten_Messungen/Hausarbeit/AUSWERTUNG_Arten.slb.hdr"
    lai_infile = "C:/GEO418/Daten_Messungen/Hausarbeit/AUSWERTUNG_LAI.slb.hdr"

    # spectral_lib.spectral(spectral_infile)

    # LAI.LAI(lai_infile)

    SM.sm_bands()
    SM.sm_lin_reg()
    SM.sm_multi_lin_reg()
    SM.pls()


if __name__ == '__main__':
    main()
