import numpy as np
import matplotlib.pyplot as plt

class TokenizeProcessor:
    def __init__(self, data_path):
        self.array = np.load(data_path).transpose((0, 3, 4, 1, 2))

    def tokenizing(self, window_size):
        output_shape = (self.array.shape[0], self.array.shape[1], self.array.shape[2], self.array.shape[3], (self.array.shape[4]-1) * pow(window_size, 2)+1)

        output_array = np.zeros(output_shape)
        padding = window_size // 2
        padded_array = np.pad(self.array, pad_width=((0,0),(padding,padding),(padding,padding),(0,0),(0,0)), mode='constant', constant_values=0)
        shape = padded_array.shape[1]
        for num_sample in range(self.array.shape[0]):
            for i in range(padding, shape-padding):
                for j in range(padding, shape-padding):
                    output_array[num_sample, i-padding, j-padding, :, :] = self.flatten_window(padded_array[num_sample, i-padding:i+padding+1, j-padding:j+padding+1, :, :], window_size)
        print(output_array.shape)
        return output_array

    def flatten_window(self, array, window_size):
        output_array = np.zeros((array.shape[2], (array.shape[3]-1)*pow(window_size,2)+1))
        for time in range(array.shape[2]):
            output_array[time, :output_array.shape[1]-1] = array[:, :, time, :5].flatten('F')
            output_array[time, output_array.shape[1]-1] = array[window_size//2, window_size//2, time, 5]
        return output_array


if __name__=='__main__':
    window_size = 5
    tokenize_processor = TokenizeProcessor('../data/proj3_train_img.npy')
    tokenized_array = tokenize_processor.tokenizing(window_size)
    np.nan_to_num(tokenized_array)
    np.save('../data/proj3_train_w'+str(window_size)+'.npy', tokenized_array.reshape(-1,10,pow(window_size,2)*5+1))
    for i in range(tokenized_array.shape[0]):
        for j in range(tokenized_array.shape[3]):
            plt.subplot(211)
            ch_label = pow(window_size,2)*5
            ch_i4 = pow(window_size,2)*3+pow(window_size,2)//2
            plt.imshow(
                (tokenized_array[i, :, :, j, ch_label] - tokenized_array[i, :, :, j, ch_label].min()) - (
                            tokenized_array[i, :, :, j, ch_label].max() - tokenized_array[i, :, :, j, ch_label].min()))
            plt.subplot(212)
            plt.imshow(
                (tokenized_array[i, :, :, j, ch_i4] - tokenized_array[i, :, :, j, ch_i4].min()) - (
                            tokenized_array[i, :, :, j, ch_i4].max() - tokenized_array[i, :, :, j, ch_i4].min()))
            plt.savefig('../plt/' + str(i) + str(j) + '.png')
            plt.show()

