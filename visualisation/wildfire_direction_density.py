import itertools
from copy import deepcopy
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def get_direction_vector(x, y, fire_t):
    '''
    Args:
        x: x-coordinate of centroid of fire pixels
        y: y-coordinate of centroid of fire pixels
        fire_t: fire pixels at timestamp t

    Returns: List of direction vectors

    '''
    vectors = []
    for i in range(fire_t.shape[0]):
        for j in range(fire_t.shape[1]):
            if fire_t[i, j] > 0:
                length = sqrt(pow(i - x, 2)+pow(j - y, 2))
                vectors.append([(j - y), -(i - x),  length])
    return vectors

def evaluate(dir):
    '''
    Args:
        dir: List of direction vectors

    Returns: maximum cosine similarity between two direction vectors.

    '''
    def evaluate_between_each_day(dir1, dir2):
        combination = []
        length = len(dir1)
        digits = [i for i in range(length)]
        for j in itertools.permutations(digits, 3):
            combination.append(j)

        score_comb = 0
        for comb in combination:
            score = 0
            for k in range(length):
                score += cosine_similarity(np.array(dir1[k]).reshape(1,-1), np.array(dir2[comb[k]]).reshape(1,-1))
            score_comb = max(score/length,score_comb)
        return score_comb
    score_avg = 0
    for i in range(1, len(dir)):
        score_avg += evaluate_between_each_day(dir[i-1], dir[i])
    return score_avg/(len(dir)-1)


if __name__=='__main__':
    # Name of testing sites
    fires = ['creek_fire', 'dixie_fire', 'lytton_fire']
    for fire in fires:
        array = np.load(fire+'.npy')
        stack = array[0, :, :]
        stack_hue = deepcopy(array[0, :, :])
        stack_hue[stack_hue==0]=np.nan
        dir = []
        plt.imshow(stack,cmap='Reds')
        plt.axis('off')
        plt.savefig('../plt/'+fire+'stack'+str(0)+'.png', bbox_inches='tight')

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

            # Visualization
            plt.figure(figsize=(20, 4))
            plt.subplot(151)
            plt.axis('off')
            plt.imshow(fire_t,cmap='Reds')
            # plt.savefig('../plt/'+fire+'fire_t'+str(i)+'.png', bbox_inches='tight')
            plt.subplot(152)
            plt.axis('off')
            plt.imshow(stack,cmap='Reds')
            plt.subplot(153)
            plt.axis('off')
            plt.imshow(stack_hue,cmap='Reds')

            plt.subplot(154)
            plt.axis('off')
            num_components = 3
            spatial_data = []
            for l in range(fire_t.shape[0]):
                for p in range(fire_t.shape[1]):
                    if fire_t[l,p] != 0:
                        spatial_data.append([l,p])

            # Kmeans Clustering of fire pixels
            if len(spatial_data)==0:
                continue
            else:
                spatial_data = np.array(spatial_data)
                cluster = KMeans(n_clusters=num_components, random_state=0).fit(spatial_data)
                cluster_center = cluster.cluster_centers_.astype(int)

            clustering_result = np.zeros((224,224))
            for k in range(len(spatial_data)):
                clustering_result[spatial_data[k][0], spatial_data[k][1]] = cluster.labels_[k]+1
            plt.scatter(spatial_data[cluster.labels_ == 0][:,1],-spatial_data[cluster.labels_ == 0][:,0], color='red')
            plt.scatter(spatial_data[cluster.labels_ == 1][:,1],-spatial_data[cluster.labels_ == 1][:,0], color='black')
            plt.scatter(spatial_data[cluster.labels_ == 2][:,1],-spatial_data[cluster.labels_ == 2][:,0], color='blue')
            plt.xlim([0, 224])
            plt.ylim([-224,0])
            plt.subplot(155)
            plt.axis('off')

            # Visualize direction vectors of active fire
            dir_day = []
            for j in range(num_components):
                arrow_x = cluster_center[j][1]-y
                arrow_y = -(cluster_center[j][0]-x)
                dir_day.append([arrow_x, arrow_y])
                plt.quiver(x, y, arrow_x, arrow_y, angles='xy', scale_units='xy', scale=1)
                plt.xlim([0, 224])
                plt.ylim([0, 224])
            dir.append(dir_day)
            plt.savefig('../plt/'+fire+'dir'+str(i)+'.png', bbox_inches='tight')
            plt.show()

            stack = np.logical_or(stack, fire_t)
            stack_hue[fire_t!=0]=i+1
            stack_hue[stack_hue==0]=np.nan
            plt.imshow(stack,cmap='Reds')
            plt.axis('off')
            plt.savefig('../plt/'+fire+'stack'+str(i)+'.png', bbox_inches='tight')

    print('Cosine Similarity of fire ' + fire+ ' {}'.format(evaluate(dir)))