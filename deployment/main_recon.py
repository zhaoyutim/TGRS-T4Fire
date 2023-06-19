import glob
import os

from matplotlib import pyplot as plt
from rasterio.crs import CRS
from rasterio.transform import Affine
os.environ['CUDA_VISIBLE_DEVICES'] = '8'
from model.vit_keras import vit
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject

def read_tiff(file_path):
    with rasterio.open(file_path) as src:
        image = src.read().transpose((1,2,0))
        metadata = src.meta
    return image, metadata

def write_tiff(file_path, arr, profile):
    with rasterio.Env():
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(arr.astype(rasterio.float32))
if __name__=='__main__':
    root_path = '/geoinfo_vol1/home/z/h/zhao2/LowResSatellitesService/data/tif_dataset/US'
    model_path = '/geoinfo_vol1/home/z/h/zhao2/proj3_models'
    tif_files = glob.glob(os.path.join(root_path, '*mosaic.tif'))
    tif_files.sort()
    results=np.load('result.npy')

    output_size_x = 224 * 46
    output_size_y = 224 * 75
    start_x = 0
    start_y = 0
    profile = {'driver': 'GTiff',
                   'dtype': 'float32',
                   'nodata': 0.0,
                   'width': output_size_y,
                   'height': output_size_x,
                   'count': 1,
                   'crs': CRS.from_epsg(4326),
                   'transform': Affine(0.00336, 0.0, -11.9+start_y*0.00336, 0.0, -0.00336, 70.51-start_x*0.00336)}

    for i in range(10):
        plt.imshow(results[:, i].reshape(output_size_x, output_size_y))
        plt.show()
    results = results.reshape((output_size, output_size, 10))
    for i in range(10):
        write_tiff('data/infer/US_'+str(i)+'.tif', results[np.newaxis, :, :, i], profile)


