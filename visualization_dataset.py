import numpy as np
from matplotlib import pyplot as plt


def standardize_img(img):
    img = img.transpose((1,2,0))
    for i in range(3):
        img[:,:,i]=(img[:,:,i]-img[:,:,i].min())/(img[:,:,i].max()-img[:,:,i].min())
    return img


if __name__=='__main__':
    path = './data/proj3_train_img_v2.npy'
    dataset = np.load(path)

    for i in range(dataset.shape[0]):
        img = dataset[i, 0, 2:5, :, :]
        img = standardize_img(img)
        plt.imshow(img)
        plt.show()