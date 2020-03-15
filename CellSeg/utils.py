from skimage import io
from astropy.io import fits


def tif2ftits(filename):
    im = io.imread(filename)

    hdu = fits.PrimaryHDU(im)
    hdu.writeto(filename.split('.')[0] + 'fits')
