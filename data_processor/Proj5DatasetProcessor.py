import copy
import datetime
import os
from datetime import timedelta
from glob import glob

import cv2
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
        df['duration'] = (pd.to_datetime(df.end_date)- pd.to_datetime(df.start_date)).dt.days
        df = df[df.duration>5]
        ids, start_dates, end_dates, lons, lats, duration = df['Id'].values.astype(str), df['start_date'].values.astype(str), df[
            'end_date'].values.astype(str), df['lon'].values.astype(float), df['lat'].values.astype(float), df['duration'].values.astype(int)
        window_size = 1
        ts_length = 10
        stack_over_location = []
        save_path = 'data/'
        n_channels = 6
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i, id in enumerate(ids):
            print(id)
            data_path = os.path.join('/Volumes/yussd/viirs/subset/', id)
            preprocessing = PreprocessingService()
            for j in range(duration[i]):
                current_date = datetime.datetime.strptime(start_dates[i], '%Y-%m-%d')+datetime.timedelta(j)
                file_list = glob(os.path.join(data_path, 'VNPIMG'+current_date.strftime('%Y-%m-%d')+'*.tif'))
                file_list.sort()
                img_list = []
                mod_list = []
                for file in file_list:
                    img_file = file
                    mod_file = file.replace('IMG', 'MOD')
                    img_array, _ = preprocessing.read_tiff(img_file)
                    mod_array, _ = preprocessing.read_tiff(mod_file)
                    img_array = cv2.resize(img_array.transpose((1,2,0)), (600,600))
                    mod_array = cv2.resize(mod_array.transpose((1,2,0)), (600,600))
                    img_list.append(img_array)
                    mod_list.append(mod_array)
                img_array = np.nanmax(np.stack(img_list, axis=0), axis=0)
                mod_array = np.nanmax(np.stack(mod_list, axis=0), axis=0)
                if os.path.exists(os.path.join('proj5_plt', id)):
                    os.mkdir(os.path.join('proj5_plt', id))
                plt.figure(figsize=(8, 4), dpi=80)
                plt.subplot(121)
                plt.imshow(self.normalization(img_array[:,:,3], False))
                plt.subplot(122)
                plt.imshow(self.normalization(np.stack([mod_array, img_array[:,:,1], img_array[:,:,0]], axis=2), False))
                plt.savefig(os.path.join('proj5_plt', id+'_'+current_date.strftime('%Y-%m-%d')+'.png'))
                plt.show()

