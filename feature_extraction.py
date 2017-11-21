import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import pickle
import cv2
import skimage
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max


def _base ( settings, params, data):
    for cur_data in data.images:
        _, _, _,nTP,n_pos = np.shape(cur_data)
        for pos in range(n_pos):
            for TP in range(nTP):
                I = cur_data[:,:,:,TP,pos]
                detect_nodes(settings,params,I)

def detect_nodes ( settings, params, I ):
    '''
    Input - I is assumed to a Z stack of a single time point - i.e., 3D image
    '''
    img = skimage.img_as_float(I)
    image_max = ndi.maximum_filter(img, size=3,mode='constant')
    coordinates = peak_local_max(img,min_distance=3)

    for i in range(np.size(img,axis=2)):


        im = img[:,:,i]
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()
        ax[0].imshow(im, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')

        ax[1].imshow(image_max[:,:,i], cmap=plt.cm.gray)
        ax[1].axis('off')
        ax[1].set_title('Maximum filter')

        ax[2].imshow(im, cmap=plt.cm.gray)
        ax[2].autoscale(False)
        idx = np.in1d(coordinates[:,2],i)
        ax[2].plot(coordinates[idx, 1], coordinates[idx, 0], 'r.')
        ax[2].axis('off')
        ax[2].set_title('Peak local max')

        fig.tight_layout()
        plt.show()
        a = 1

