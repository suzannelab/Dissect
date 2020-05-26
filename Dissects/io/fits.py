from skimage import io
from astropy.io import fits


def tif2fits(filename):
    im = io.imread(filename)

    hdu = fits.PrimaryHDU(im)
    # modification de l'header pour avoir la conversion µm/pixel
    #fits.setval(hdu, 'pixel_width', value='M31')
    #fits.setval(hdu, 'pixel_height', value='M31')
    #fits.setval(hdu, 'voxel_depth', value='M31')
    hdu.writeto(filename.split('.')[0] + '.fits')