"""
script for preprocessing the raw WSIs into 1536 tiles un-normalized or normalized tiles
"""
import sys 
import pyvips as Vips
import os
import numpy as np
from tqdm import tqdm
import vips_utils, normalize

def tileRawWSIToNormalized1536():
    """
    Takes the WSIs found in WSI_DIR, and tiles them to color-normalized 1536 x 1536 pixels
    Will print error messages if a WSI fails to tile (this will happen if CPU is not powerful enough, or too many concurrent threads running on the system)
    """
    ref_imagenames = {}
    ref_imagenames['4G8'] = 'NA4996UCDtemporalgyri _4G8.svs'
    ref_imagenames['Biels'] = 'NA4092UCDTemporalgyri_Biels.svs'
    ref_imagenames['6E10'] = '50522-3_UTSW_temporal_6E10.svs'
    ref_imagenames['Kofler'] = 'cw18-015.svs'
    WSI_DIR = 'data/raw_WSIs/' 
    SAVE_DIR = 'data/normalized_tiles/'
    wsi_files = os.listdir(WSI_DIR)
    imagenames = sorted(wsi_files)
    stains = {}
    stains['4G8'] = [image for image in imagenames if image.endswith('_4G8.svs')]
    stains['Biels'] = [image for image in imagenames if image.endswith('_Biels.svs')]
    stains['6E10'] = [image for image in imagenames if image.endswith('_6E10.svs')]
    stains['Kofler'] = [image for image in imagenames if image.startswith('cw')]
    for stain in stains:
        if len(stains[stain]) == 0:
            continue
        ref_image = Vips.Image.new_from_file(WSI_DIR + ref_imagenames[stain], level=0)
        normalizer = normalize.Reinhard()
        normalizer.fit(ref_image)
        failed_images = []
        for imagename in stains[stain]:
            print(imagename)
            vips_img = Vips.Image.new_from_file(WSI_DIR + imagename, level=0)
            if isImage40x(imagename):
                print("resizing 40x to 20x")
                vips_img = vips_img.resize(0.5)
                print("resized")
            out = normalizer.transform(vips_img)
            out.filename = vips_img.filename
            try:
                vips_utils.save_and_tile(out, SAVE_DIR)
                print("saved and tiled")
            except:
                print("could not tile: {}".format(imagename))
                failed_images.append(imagename)
        print("failed to tile: {}".format(failed_images))

def isImage40x(img_name):
    """
    will return True if img_name belongs to a WSI that is 40x native 
    """
    specials = ['NA5012UCDtemporalgyri _4G8', 'NA5013UCDtemporalgyri _4G8', 'NA4963UCDtemporalgyri _4G8', 'NA5009UCDtemporalgyri _4G8'] ##mismatched 1536 tiles between norm and unnorm, NA5009 and NA5013 are WSIs used for phase 2 though 
    for img in specials:
        if img in img_name:
            return True
    return False

tileRawWSIToNormalized1536()

