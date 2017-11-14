import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import pickle
import cv2
import skimage


def _base ( settings, params, data):
    for cur_data in data.images:
        _, _, _,nTP,n_pos = np.shape(cur_data)
        for pos in range(n_pos):
            for TP in range(nTP):
                I = cur_data[:,:,:,TP,pos]
                skeletonize_image(settings,params,I)

def skeletonize_image ( settings, params, I ):
    '''
    Input - I is assumed to a Z stack of a single time point - i.e., 3D image
    '''
    a = skimage.skeletonize(I[:,:,0])