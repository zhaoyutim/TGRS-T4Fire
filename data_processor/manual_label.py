import numpy as np
import matplotlib.pyplot as plt
def manual_label(array):
    th = 0.1
    for i in range(array.shape[0]):
        img = (array[i,3,:,:]-array[i,4,:,:])/(array[i,3,:,:]+array[i,4,:,:])
        plt.subplot(121)
        plt.imshow(img>th)
        plt.subplot(122)
        plt.imshow(array[i, 5, :, :])
        plt.show()
    return array


if __name__=='__main__':
    th = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    window_size = 3
    array = np.load('../data/proj3_test_img.npy')
    print(array.shape)
    array = manual_label(array)

