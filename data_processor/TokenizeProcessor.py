import platform

import numpy as np

class TokenizeProcessor:

    def tokenizing(self, data_path, window_size):
        array = np.load(data_path).transpose((0, 3, 4, 1, 2))
        return array
    def flatten_window(self, array, window_size):
        output_array = np.zeros((array.shape[2], (array.shape[3])*pow(window_size,2)))
        for time in range(array.shape[2]):
            output_array[time, :] = array[:, :, time, :].flatten('F')
        return output_array



if __name__=='__main__':
    import os
    if platform.system() == 'Darwin':
        root_path = '/Users/zhaoyu/PycharmProjects/T4Fire/data'
    else:
        root_path = '/geoinfo_vol1/zhao2/proj5_dataset'
    window_size = 1
    ts_length=10
    tokenize_processor = TokenizeProcessor()
    tokenized_array = tokenize_processor.tokenizing(os.path.join(root_path,'proj5_train_img_seqtoseq_l'+str(ts_length)+'.npy'), window_size)
    np.nan_to_num(tokenized_array)
    np.save(os.path.join(root_path,'proj5_train_img_seqtoone_l'+str(ts_length)+'_w'+str(window_size)+'.npy'), tokenized_array.reshape(-1,tokenized_array.shape[-2],tokenized_array.shape[-1]))

