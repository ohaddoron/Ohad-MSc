import numpy as np
import skimage
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import exposure
from scipy.signal import medfilt2d

def _base ( settings, params, data):
    for cur_data in data.images:
        _, _,_,nTP = np.shape(cur_data)
        for TP in range(nTP):
                I = cur_data[:,:,:,TP]
                nodes,coordinates = extract_nodes(settings,params,I)
                a = 1

def extract_nodes ( settings, params, I ):
    '''
    Input - I is assumed to a Z stack of a single time point - i.e., 3D image
    '''
    img = skimage.img_as_float(I)
    nodes = ()
    for i in range(np.size(img,axis=2)):


        im_adjusted = exposure.equalize_hist(img[:,:,i])
        im_adjusted = medfilt2d(im_adjusted,kernel_size=params.med_filter_kernel_size)
        image_max = ndi.maximum_filter(im_adjusted, size=params.max_filter_kernel_size, mode='constant')
        coordinates = peak_local_max(im_adjusted, min_distance=params.peak_detector_size)
        #coordinates = remove_artifacts(im_adjusted,coordinates)

        coordinates = coordinates_distance(params, coordinates)

        '''
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        
        ax = axes.ravel()
        ax[0].imshow(img[:,:,i], cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')

        ax[1].imshow(image_max, cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Maximum filter')

        ax[2].imshow(img[:,:,i], cmap=plt.cm.gray)
        ax[2].autoscale(False)
        #idx = np.in1d(coordinates[:,2],i)
        ax[2].scatter(coordinates[:, 1], coordinates[:, 0],facecolors = 'none',edgecolors='r')
        ax[2].axis('off')
        ax[2].set_title('Peak local max')

        fig.tight_layout()
        plt.show()
        '''


        for k in range(np.shape(coordinates)[0]):
            min_x = coordinates[k, 0] - params.node_size
            max_x = coordinates[k, 0] + params.node_size
            min_y = coordinates[k, 1] - params.node_size
            max_y = coordinates[k, 1] + params.node_size
            width,height = np.shape(im_adjusted)
            if min_x > 0 and min_y > 0 and max_x < int(width) and max_y < int(height):
                nodes += (img[min_x:max_x,min_y:max_y,i],)
    return nodes,coordinates


def remove_artifacts ( I,coordinates ):
    i = 0
    coordinates = list(coordinates)
    while i < np.shape(coordinates)[0]:
        if I[coordinates[i][0],coordinates[i][1]] > 0.98:
            coordinates.pop(i)
        else:
            i+=1
    return np.asanyarray(coordinates)
def coordinates_distance ( params,coordinates ):
    num_coordinates = np.shape(coordinates)[0]
    i = 0
    j = 0

    while i < num_coordinates:
        tmp = coordinates
        count = 0
        j = 0
        while j < num_coordinates:
            dist = np.linalg.norm(coordinates[i,:] - coordinates[j,:])
            if dist < params.dist_threshold and dist > 0:
                tmp = np.delete(tmp, (count), axis=0)
                num_coordinates -= 1
            else:
                count += 1
            j +=1
        coordinates = tmp
        i+=1
    return coordinates


