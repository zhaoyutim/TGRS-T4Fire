import gc
import glob
import os
from rasterio.crs import CRS
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow_addons as tfa
os.environ["TF_ENABLE_MLIR_OPTIMIZATIONS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '8, 9'
import tensorflow as tf
import numpy as np
import rasterio

output_size_x = 224 * 15
output_size_y = 224 * 15
start_x = 3000
start_y = 0
def read_tiff(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()
        # plt.imshow(image[3,7700:7700+595,1400:1400+595])
        # plt.show()
        image = standardization_tile(image[:,start_y:start_y+output_size_y ,start_x:start_x+output_size_x]).transpose((1,2,0))
        metadata = src.meta
    return image, metadata

def write_tiff(file_path, arr, profile):
    with rasterio.Env():
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(arr.astype(rasterio.float32))

def standardization(array):
    n_channels = array.shape[0]
    nanmean = [17.46, 26.88, 19.99, 295.6, 277.9]
    std = [16.35, 16.94, 12.48, 6.54, 8.5]
    for i in range(n_channels):
        # nanmean = np.nanmean(array[i, :, :])
        array[i, :, :] = np.nan_to_num(array[i, :, :], nan=nanmean[i])
        array[i,:,:] = (array[i,:,:]-nanmean[i])/std[i]
    return np.nan_to_num(array)

def standardization_tile(array):
    n_channels = array.shape[0]
    x = array.shape[1]
    y = array.shape[2]
    for i in range(n_channels):
        for j in range(x//224):
            for k in range(y//224):
                nanmean = np.nanmean(array[i, j*224:(j+1)*224, k*224:(k+1)*224])
                nanstd = np.nanstd(array[i, j*224:(j+1)*224, k*224:(k+1)*224])
                array[i, j*224:(j+1)*224, k*224:(k+1)*224] = np.nan_to_num(array[i, j*224:(j+1)*224, k*224:(k+1)*224], nan=nanmean)
                array[i,j*224:(j+1)*224, k*224:(k+1)*224] = (array[i,j*224:(j+1)*224, k*224:(k+1)*224]-nanmean)/(nanstd+1e-6)
    return np.nan_to_num(array)
class TQDMPredictCallback(tf.keras.callbacks.Callback):
    def __init__(self, custom_tqdm_instance=None, tqdm_cls=tqdm, **tqdm_params):
        super().__init__()
        self.tqdm_cls = tqdm_cls
        self.tqdm_progress = None
        self.prev_predict_batch = None
        self.custom_tqdm_instance = custom_tqdm_instance
        self.tqdm_params = tqdm_params

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self.tqdm_progress.update(batch - self.prev_predict_batch)
        self.prev_predict_batch = batch

    def on_predict_begin(self, logs=None):
        self.prev_predict_batch = 0
        if self.custom_tqdm_instance:
            self.tqdm_progress = self.custom_tqdm_instance
            return

        total = self.params.get('steps')
        if total:
            total -= 1

        self.tqdm_progress = self.tqdm_cls(total=total, **self.tqdm_params)

    def on_predict_end(self, logs=None):
        if self.tqdm_progress and not self.custom_tqdm_instance:
            self.tqdm_progress.close()

if __name__=='__main__':
    id = 'CANADA'
    root_path = '/geoinfo_vol1/home/z/h/zhao2/LowResSatellitesService/data/tif_dataset/'+id
    model_path = '/geoinfo_vol1/home/z/h/zhao2/proj3_models'
    tif_files = glob.glob(os.path.join(root_path, '*mosaic.tif'))
    tif_files.sort()
    # tif_files = tif_files[7:17]
    arrays=[]
    for i in range(10):
        array, _ = read_tiff(tif_files[i])
        array = array.reshape((-1, 5))
        arrays.append(array)
    fire_dataset = np.stack(arrays, axis=1)
    x_min = -138
    y_max = 60
    profile = {'driver': 'GTiff',
                   'dtype': 'float32',
                   'nodata': 0.0,
                   'width': output_size_x,
                   'height': output_size_y,
                   'count': 1,
                   'crs': CRS.from_epsg(4326),
                   'transform': Affine(0.00336, 0.0, x_min+start_x*0.00336, 0.0, -0.00336, y_max-start_y*0.00336)}

    del arrays
    MAX_EPOCHS = 50
    window_size = 1
    learning_rate = 0.001
    weight_decay = 0.0001
    proj_dim = pow(window_size, 2) * 5
    num_classes = 2
    input_shape = (10, proj_dim)
    load_pretrained = True
    pretrained = 'nopretrained'
    model_name = 'vit_tiny_custom'
    run = 'run1'
    num_heads = 3
    mlp_dim = 112
    num_layers = 4
    hidden_size = 24
    batch_size = 512
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    print(fire_dataset.shape)
    results = np.zeros((fire_dataset.shape[0],fire_dataset.shape[1]))
    slice_size = 10000000
    loop_size = fire_dataset.shape[0]//slice_size
    logger = TQDMPredictCallback()
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    with strategy.scope():
        model = tf.keras.models.load_model('/geoinfo_vol1/zhao2/proj3_models' + '/proj3_' + model_name + 'w' + str(
            window_size) + '_nopretrained' + '_' + str(run) + '_' + str(num_heads) + '_' + str(mlp_dim) + '_' + str(
            hidden_size) + '_' + str(num_layers) + '_' + str(batch_size), custom_objects={"AdamW": optimizer})

        for i in range(loop_size):
            print(i, loop_size)
            results[i*slice_size:(i+1)*slice_size,:] = model.predict(fire_dataset[i*slice_size:(i+1)*slice_size,:,:], batch_size=512, callbacks=[logger])[:, :, 1] > 0.5
            gc.collect()
        results[(loop_size) * slice_size:, :] = model.predict(fire_dataset[loop_size * slice_size:, :, :], batch_size=512, callbacks=[logger])[:, :, 1] > 0.5
        gc.collect()
    # del fire_dataset
    np.save('result.npy', results)
    for i in range(10):
        figure, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(12, 5))
        ax[0].imshow(fire_dataset[:, i, 3].reshape(output_size_y, output_size_x))
        ax[1].imshow(results[:,i].reshape(output_size_y, output_size_x))
        plt.savefig('results/'+id+str(i)+'.png', bbox_inches='tight')
        plt.show()
    results = results.reshape((output_size_y, output_size_x, 10))
    for i in range(10):
        write_tiff('datainfer/'+tif_files[i].split('/')[-1][6:16]+'.tif', results[np.newaxis, :, :, i], profile)