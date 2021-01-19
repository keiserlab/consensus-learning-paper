"""
script containing utility functions, written by Ziqi Tang
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvips as Vips

NP_DTYPE_TO_VIPS_FORMAT = {
        np.dtype('int8'): Vips.BandFormat.CHAR,
        np.dtype('uint8'): Vips.BandFormat.UCHAR,
        np.dtype('int16'): Vips.BandFormat.SHORT,
        np.dtype('uint16'): Vips.BandFormat.USHORT,
        np.dtype('int32'): Vips.BandFormat.INT,
        np.dtype('float32'): Vips.BandFormat.FLOAT,
        np.dtype('float64'): Vips.BandFormat.DOUBLE
    }

VIPS_FORMAT_TO_NP_DTYPE = {v:k for k, v in NP_DTYPE_TO_VIPS_FORMAT.items()}

class Reinhard(object):
    """
    A stain normalization object for PyVips.
    fits a reference PyVips image,
    transforms a PyVips Image.
    Can also be initialized with precalculated
    means and stds (in LAB colorspace)
    """

    def __init__(self, target_means=None, target_stds=None):
        self.target_means = target_means
        self.target_stds  = target_stds

    def fit(self, target): 
        """
        target is a PyVips Image object
        """
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds  = stds
    
    def transform(self, image):
        L, A, B = self.lab_split(image)
        means, stds = self.get_mean_std(image)
        norm1 = ((L - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((A - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((B - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self.merge_to_rgb(norm1, norm2, norm3)
    
    def lab_split(self, img):
        img_lab = img.colourspace("VIPS_INTERPRETATION_LAB")
        L, A, B = img_lab.bandsplit()[:3]
        return L, A, B
        
    def get_mean_std(self, image):
        L, A, B = self.lab_split(image)
        m1, sd1 = L.avg(), L.deviate()
        m2, sd2 = A.avg(), A.deviate()
        m3, sd3 = B.avg(), B.deviate()
        means = m1, m2, m3
        stds  = sd1, sd2, sd3
        self.image_stats = means, stds
        return means, stds
    
    def merge_to_rgb(self, L, A, B):
        img_lab = L.bandjoin([A,B])
        img_rgb = img_lab.colourspace('VIPS_INTERPRETATION_sRGB')
        return img_rgb

def array_vips(vips_image, verbose=False):
    # dtype = np.dtype('u{}'.format(vips_image.BandFmt.bit_length() + 1))
    dtype = VIPS_FORMAT_TO_NP_DTYPE[vips_image.format]
    if verbose:
        print(dtype, vips_image.height, vips_image.width, vips_image.bands)
    return (np.fromstring(vips_image.write_to_memory(), dtype=dtype) #np.uint8)
             .reshape(vips_image.height, vips_image.width, vips_image.bands))

def show_vips(vips_image, ax=plt, show=True, verbose=False):
    if not isinstance(vips_image, Vips.Image):
        return -1
    
    im_np = array_vips(vips_image)
    if verbose:
        print(im_np.shape)
    if vips_image.bands == 1:
        ax.imshow(im_np.squeeze()/np.max(im_np), cmap=plt.get_cmap('gist_ncar'))
    elif vips_image.bands == 2:
        im_np = im_np[:,:,1]
        ax.imshow(im_np/np.max(im_np), cmap=plt.get_cmap('gray'))
    else:
        ax.imshow(im_np)
    if show:
        plt.show()
    
def image_fields_dict(im_with_fields):
    return {k:im_with_fields.get(k) 
            for k in im_with_fields.get_fields() 
            if im_with_fields.get_typeof(k)}
# from https://github.com/jcupitt/libvips/blob/master/doc/Examples.md

NP_DTYPE_TO_VIPS_FORMAT = {
        np.dtype('int8'): Vips.BandFormat.CHAR,
        np.dtype('uint8'): Vips.BandFormat.UCHAR,
        np.dtype('int16'): Vips.BandFormat.SHORT,
        np.dtype('uint16'): Vips.BandFormat.USHORT,
        np.dtype('int32'): Vips.BandFormat.INT,
        np.dtype('float32'): Vips.BandFormat.FLOAT,
        np.dtype('float64'): Vips.BandFormat.DOUBLE
    }

VIPS_FORMAT_TO_NP_DTYPE = {v:k for k, v in NP_DTYPE_TO_VIPS_FORMAT.items()}

def array_vips(vips_image, verbose=False):
    # dtype = np.dtype('u{}'.format(vips_image.BandFmt.bit_length() + 1))
    dtype = VIPS_FORMAT_TO_NP_DTYPE[vips_image.format]
    if verbose:
        print(dtype, vips_image.height, vips_image.width, vips_image.bands)
    return (np.fromstring(vips_image.write_to_memory(), dtype=dtype) #np.uint8)
             .reshape(vips_image.height, vips_image.width, vips_image.bands)).squeeze()

def show_vips(vips_image, ax=plt, show=True, verbose=False):
    if not isinstance(vips_image, Vips.Image):
        return -1
    
    im_np = array_vips(vips_image)
    if verbose:
        print(im_np.shape)
    if vips_image.bands == 1:
        ax.imshow(im_np/np.max(im_np), cmap=plt.get_cmap('gist_ncar'))
    elif vips_image.bands == 2:
        im_np = im_np[:,:,1]
        ax.imshow(im_np/np.max(im_np), cmap=plt.get_cmap('gray'))
    else:
        ax.imshow(im_np)
    if show:
        plt.show()
    
def image_fields_dict(im_with_fields):
    return {k:im_with_fields.get(k) 
            for k in im_with_fields.get_fields() 
            if im_with_fields.get_typeof(k)}

def save_and_tile(image_to_segment, output_dir, tile_size=1536):
    basename = os.path.basename(image_to_segment.filename)
    base_dir_name = os.path.join(output_dir, basename.split('.svs')[0])
    print("base dir name", base_dir_name)
    if not os.path.exists(base_dir_name):
        os.makedirs(base_dir_name)
    Vips.Image.dzsave(image_to_segment, base_dir_name,
                        layout='google',
                        suffix='.jpg[Q=90]',
                        tile_size=tile_size,
                        depth='one',
                        properties=True)
    return None




