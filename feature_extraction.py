import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import scipy.ndimage.filters as filters
from scipy.io import savemat
from tqdm import tqdm
# %%

def _base ( settings, params, data):
    '''
    :param settings: general settings used
    :param params: general parameters used
    :param data: data loaded from file
    :return:
    '''

    for i in range(len(data['images'])): # the search is performed for each image individually (for the first time point)
        img = data['images'][i]
        # img -= np.mean(img) # subtract the mean of the image to start

        # templates = generate_templates(params,img)
        coordinates = detect_peaks(img)
        # seed = [[x[5][3],y[5][3],5L]]
        # region_growing_3D(params,img,seed)


        MASKS = region_growing_3D(params,img,coordinates)

        flag = 1




def region_growing_3D ( params, img, seeds,intensity_threshold=0.04,distance_threshold=20 ):
    '''
    :param params: parameter structure
    :param img: image to perform region growing on
    :param seed: seed to start the region growing algorithm
    :return: coordinates of blob matching current seed
    '''
    seeds = list(seeds)
    MASKS = ()
    for seed in tqdm(seeds):
        # seed = seeds.pop()
        Q = [seed]
        r, c, p = np.shape(img)
        J = np.zeros((r, c, p))
        regVal = img[seed[0],seed[1],seed[2]]
        while Q:
            cur_seed = Q.pop()
            xv = cur_seed[0];
            yv = cur_seed[1];
            zv = cur_seed[2]
            J[xv,yv,zv] = 1
            for i in range(-1,1):
                for j in range(-1,1):
                    for k in range(-1,1):
                        if (xv + i >= 0 and xv+1 < r
                            and yv + j >= 0 and yv + j < c
                            and zv + k >= 0 and zv + k < p
                            and any([i,j,k])
                            and J[xv+i,yv+j,zv+k] == 0
                            and img[xv+i,yv+j,zv+k] <= regVal + intensity_threshold
                            and img[xv+i,yv+j,zv+k] >= regVal - intensity_threshold
                            and np.sqrt((seed[0] - xv+i)**2 + (seed[1]-yv+j)**2 + (seed[2]-zv+k)**2) <= distance_threshold):
                                J[xv+i,yv+j,zv+k] = 1
                                Q.append([xv+i,yv+j,zv+k])
                                regVal = np.mean(img[np.where(J==1)])
        for coord in np.transpose(np.where(J)):
            idx2remove = np.where([np.array_equal(coord,sub_seed) for sub_seed in seeds])
            if len(idx2remove[0]) > 0:
                seeds.pop(idx2remove[0])

        # if np.shape(np.where(J))[1] > 50:
        MASKS += (np.transpose(np.where(J)),)
    np.save('../results/DIST_THRESH=%2.2f_INTENSITY_THRESH=%2.2f' % (distance_threshold,intensity_threshold),MASKS)
    return MASKS


def detect_peaks(image,neighborhood_size=20):

    # image_max = ndi.maximum_filter(image, size=neighborhood_size, mode='constant')
    for plane in range(image.shape[-1]):
        tmp_coordinates = peak_local_max(image[:,:,plane], min_distance=20)
        tmp_coordinates = [np.concatenate((coord, [plane])) for coord in tmp_coordinates]
        if 'coordinates' not in locals():
            coordinates = tmp_coordinates
        else:
            coordinates = np.vstack((coordinates,tmp_coordinates))

    return coordinates

def generate_templates ( params,img ):
    '''
    :param params: general parameters used
    :param img: 3D image to be analyzed
    :return: templates

    Templates are generated based on Parameter-Free Binarization and Skeletonization of Fiber Networks from Confocal Image Stacks
    The searching axis will always be 0-1 so the input image will have to be transposed for the axis to change from one
    iteration to the next.
    '''



    shape = np.shape(img)

    template_size_x = int(shape[0] * params.template_size_x) # initial template size
    template_size_y = int(shape[1] * params.template_size_y) # initial template size
    template_size_z = int(shape[2] * params.template_size_z)  # initial template size
    x_y_template = extract_template(params,img,template_size_x,template_size_y)
    x_z_template = extract_template(params,np.transpose(img,[0,2,1]),template_size_x,template_size_z)
    y_z_template = extract_template(params,np.transpose(img,[1,2,0]),template_size_y,template_size_z)
    templates = {'x-y': x_y_template, 'x-z' : x_z_template, 'y-z' : y_z_template}
    return templates


def extract_template ( params , img,template_size_0,template_size_1 ):
    '''
    :param params: general class of parameters used
    :param img: image that we wish to extract volumes from
    :param sample_idx: linear indices of voxels that will be the center of the extraction process
    :param template_size_0: template size in the first direction
    :param template_size_1: template size in the second direction
    :return: extracted volumes
    '''
    # The third axis is always used as an anchor and nothing is extracted around it
    # %% init
    count = 0
    slicing_indices = []
    flag = True

    while flag:
        random.seed(params.seed)
        sample_idx = random.sample(list(range(img.size)),
                                   params.num_nodes_generate)  # draw indices that we will later use to search in
        for i in range(params.num_nodes_generate):
            slicing_indices.append(np.unravel_index(sample_idx[i],img.shape))
        # %%


        for i,indices in enumerate(slicing_indices):
            min0 = indices[0] - template_size_0
            max0 = indices[0] + template_size_0
            min1 = indices[1] - template_size_1
            max1 = indices[1] + template_size_1

            if min0 < 0 or min1 < 0 or max0 > np.shape(img)[0] or max1 > np.shape(img)[1]:
                # in case the requested indices are out of the image, continue so we won't attempt
                # to extract templates of the wrong size
                continue

            cur_template = img[min0:max0,min1:max1,indices[2]]/img[indices]
            count += 1
            if i == 0:
                template = cur_template
            else:
                template += cur_template


            # if i == 0:
            #     sample = img[min_x:max_x,min_y:max_y,min_z:max_z]

        template = template / count # convert the sum to average

        # validate template
        validation_set = np.concatenate((template[:,0],template[:,-1],
                                         np.transpose(template[0,:]),np.transpose(template[-1,:])),axis=0)
        if np.mean(validation_set < 0) > 0.7: # if we have too many negative values
            # on the edges and we have not increase in the previous iteration
            template_size_0 -= 1
            template_size_1 -= 1
        elif np.mean(validation_set > 0) > 0.7: # if we have too many positive
            # values on the edges and we have not decreased in the previous iteration
            template_size_0 += 1
            template_size_1 += 1
        else:
            return template



