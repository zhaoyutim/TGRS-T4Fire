import copy
import os
from datetime import timedelta
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rasterio._io import Affine

from data_processor.PreprocessingService import PreprocessingService

class Proj5DatasetProcessor(PreprocessingService):
    def dataset_generator_proj5_image(self, year, file_name, image_size=(224, 224)):
        filename = 'roi/us_fire_' + year + '_out.csv'
        df = pd.read_csv(filename)
        df.start_date = pd.to_datetime(df.start_date)
        df.end_date = pd.to_datetime(df.end_date)
        df['duration'] = (df.end_date- df.start_date).dt.days
        df = df[df.duration>5]
        ids, start_dates, end_dates, lons, lats = df['Id'].values.astype(str), df['start_date'].values.astype(str), df[
            'end_date'].values.astype(str), df['lon'].values.astype(float), df['lat'].values.astype(float)
        window_size = 1
        ts_length = 10
        stack_over_location = []
        save_path = 'data/'
        n_channels = 6
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for id in ids:
            print(id)
            data_path = os.path.join('/Volumes/yussd/viirs/subset/', id)
            file_list = glob(os.path.join(data_path, 'VNPIMGPRO*.tif'))
            file_list.sort()
            preprocessing = PreprocessingService()
            for file in file_list:
                img_file = file
                mod_file = file.replace('IMG', 'MOD')

                img_array, _ = preprocessing.read_tiff(img_file)
                mod_array, _ = preprocessing.read_tiff(mod_file)

                plt.figure(figsize=(8, 4), dpi=80)
                plt.subplot(121)
                plt.imshow(self.normalization(img_array[:3,:,:]).transpose((1,2,0)))
                plt.subplot(122)
                plt.imshow(self.normalization(mod_array).transpose((1,2,0)))
                plt.show()

