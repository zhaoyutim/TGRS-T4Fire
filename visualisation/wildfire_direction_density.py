from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from sklearn.cluster import KMeans


def get_direction_vector(x, y, fire_t):
    vectors = []
    vector = [0, 0]
    for i in range(fire_t.shape[0]):
        for j in range(fire_t.shape[1]):
            if fire_t[i, j] > 0:
                length = sqrt(pow(i - x, 2)+pow(j - y, 2))
                vectors.append([(j - y), -(i - x),  length])
    return vectors


def get_intensity(fire_t, kernal):
    return scipy.signal.convolve2d(fire_t, kernal, mode='same')


def get_clusters(spatial_data, num_components):
    kmeans = KMeans(n_clusters=num_components, random_state=0).fit(spatial_data)
    return kmeans


if __name__=='__main__':
    fires = ['creek_fire', 'dixie_fire', 'lytton_fire']
    for fire in fires:
        array = np.load(fire+'.npy')
        stack = array[0, :, :]
        for i in range(1, 10):
            fire_t = array[i, :, :]
            # Calculate centroid of the fire
            index = np.where(stack>0)
            if index[0].size != 0:
                x = int(index[0].mean())
                y = int(index[1].mean())

                vectors = get_direction_vector(x, y, fire_t)
            else:
                vectors = [[0, 0]]
            plt.figure(figsize=(16, 4))
            plt.subplot(141)
            plt.axis('off')
            plt.imshow(fire_t,cmap='Reds')
            # plt.savefig('../plt/'+fire+'fire_t'+str(i)+'.png', bbox_inches='tight')
            plt.subplot(142)
            plt.axis('off')
            plt.imshow(stack,cmap='Reds')
            # plt.savefig('../plt/'+fire+'stack'+str(i)+'.png', bbox_inches='tight')
            # plt.subplot(233)
            # center = np.zeros((224,224))
            # center[x,y]=1
            # plt.imshow(center)
            # plt.subplot(234)
            # for vec in vectors:
            #     plt.quiver(x, y, vec[0], vec[1], angles='xy', scale_units='xy', scale=1)
            #     plt.xlim([0, 224])
            #     plt.ylim([0, 224])
            # plt.subplot(235)
            # kernal = np.ones((10,10))
            # plt.imshow(get_intensity(fire_t, kernal), cmap='hot', interpolation='nearest')
            # plt.colorbar()

            plt.subplot(143)
            plt.axis('off')
            num_components = 3
            spatial_data = []
            for l in range(fire_t.shape[0]):
                for p in range(fire_t.shape[1]):
                    if fire_t[l,p] != 0:
                        spatial_data.append([l,p])
            spatial_data = np.array(spatial_data)

            cluster = get_clusters(spatial_data, num_components)
            cluster_center = cluster.cluster_centers_.astype(int)
            clustering_result = np.zeros((224,224))

            for k in range(len(spatial_data)):
                clustering_result[spatial_data[k][0], spatial_data[k][1]] = cluster.labels_[k]+1
            plt.scatter(spatial_data[cluster.labels_ == 0][:,1],-spatial_data[cluster.labels_ == 0][:,0], color='red')
            plt.scatter(spatial_data[cluster.labels_ == 1][:,1],-spatial_data[cluster.labels_ == 1][:,0], color='black')
            plt.scatter(spatial_data[cluster.labels_ == 2][:,1],-spatial_data[cluster.labels_ == 2][:,0], color='blue')
            plt.xlim([0, 224])
            plt.ylim([-224,0])
            # plt.savefig('../plt/'+fire+'clustering'+str(i)+'.png', bbox_inches='tight')
            plt.subplot(144)
            plt.axis('off')
            for j in range(num_components):
                arrow_x = cluster_center[j][1]-y
                arrow_y = -(cluster_center[j][0]-x)
                plt.quiver(x, y, arrow_x, arrow_y, angles='xy', scale_units='xy', scale=1)
                plt.xlim([0, 224])
                plt.ylim([0, 224])
            plt.savefig('../plt/'+fire+'dir'+str(i)+'.png', bbox_inches='tight')
            plt.show()

            stack = np.logical_or(stack, fire_t)
