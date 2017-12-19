import numpy as np
import skimage
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import exposure
from scipy.signal import medfilt2d
from tqdm import tqdm
import cv2
import imutils
# %% 
def _base ( settings, params, data):


    nodes,coordinates = extract_nodes(settings,params,data)
    match_nodes(nodes,coordinates,data)
# %% 
def match_nodes ( nodes,coordinates,data ):
    nTP = len(nodes)
    m_score = ()
    z_score = ()
    for TP in range(nTP - 1):
        node_detector(nodes[TP],coordinates,data.images[0][:,:,:,TP+1])
        # tmp_m,tmp_z = node_scores(nodes[TP],nodes[TP+1])
        # m_score += (tmp_m,)
        # z_score += (tmp_z,)
    return m_score,z_score
# %%
def node_detector ( nodes,coords, Z_Stack ):
    for node in nodes:
        for k in range(Z_Stack.shape[2]):
            I = np.round(Z_Stack[:,:,k]/np.max(Z_Stack[:,:,k]) * 255)
            I = I.astype(np.uint8)
            node = np.round(node / np.max(node) * 255)
            node = node.astype(np.uint8)
            res = cv2.matchTemplate(Z_Stack[:,:,k].astype(np.uint8),node.astype(np.uint8),cv2.TM_CCOEFF_NORMED )
            w,h = node.shape[:2]
            (_, _, minLoc, maxLoc) = cv2.minMaxLoc(res)

            topLeft = maxLoc
            botRight = (topLeft[0] + w, topLeft[1] + h)
            roi = I[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]
            mask = np.zeros(I.shape, dtype="uint8")
            I = cv2.addWeighted(I, 0.25, mask, 0.75, 0)
            I[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
            cv2.imshow("I", imutils.resize(I, height=650))
            cv2.imshow("node", node)

            # threshold = 0.8
            # loc = np.where(res >= threshold)
            a = 1

# %%
def node_scores ( nodes1, nodes2 ):
    m_score = np.zeros((len(nodes1),len(nodes2)))
    z_score = np.zeros((len(nodes1),len(nodes2)))
    for i,node1 in enumerate(nodes1):
        for j,node2 in enumerate(nodes2):
            pass

    return m_score,z_score

# %% 
def extract_nodes ( settings,params,data):
    '''init'''
    nodes = () # Initialize nodes and gather them in tuple
    coordinates = () # Initialize coordinates and gather them in tuple
    try: # attempt to find how many time points exist
        _,_,_,nTP = np.shape(data.images[0]) # number of time points in file; # currently assuming a single file
    except: # if only a single time point exists
        data.images = (np.expand_dims(data.images[0],axis=4),)
        nTP = 1

    for TP in tqdm(range(nTP)):
        '''TP cycle'''
        I = data.images[0][:,:,:,TP]
        tmp_nodes,tmp_coordinates = extract_nodes_single_TP(settings,params,I)
        nodes += (tmp_nodes,)
        coordinates += (tmp_coordinates,)
    return nodes,coordinates

# %% 
def extract_nodes_single_TP ( settings, params, I ):
    '''
    Input - I is assumed to a Z stack of a single time point - i.e., 3D image
    '''
    img = skimage.img_as_float(I)
    nodes = ()
    for i in range(np.size(img,axis=2)):


        im_adjusted = exposure.equalize_hist(img[:,:,i])
        im_adjusted = medfilt2d(im_adjusted,kernel_size=params.med_filter_kernel_size)
        image_max = ndi.maximum_filter(img, size=params.max_filter_kernel_size, mode='constant')
        coordinates = peak_local_max(image_max, min_distance=params.peak_detector_size)
        #coordinates = remove_artifacts(im_adjusted,coordinates)

        # coordinates = coordinates_distance(params, coordinates)

        if settings.visualize_node_detection:
            fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True,
                                     subplot_kw={'adjustable': 'box-forced'})
            ax = axes.ravel()
            ax[0].imshow(img[:,:,i], cmap=plt.cm.gray)
            ax[0].axis('off')
            ax[0].set_title('Original')

            ax[1].imshow(image_max[:,:,i], cmap=plt.cm.gray)
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



        for k in range(np.shape(coordinates)[0]):
            min_x = coordinates[k, 0] - params.node_size
            max_x = coordinates[k, 0] + params.node_size
            min_y = coordinates[k, 1] - params.node_size
            max_y = coordinates[k, 1] + params.node_size
            width,height = np.shape(im_adjusted)
            if min_x > 0 and min_y > 0 and max_x < int(width) and max_y < int(height):
                nodes += (img[min_x:max_x,min_y:max_y,i],)
    return nodes,coordinates


# %%
def remove_artifacts ( I,coordinates ):
    i = 0
    coordinates = list(coordinates)
    while i < np.shape(coordinates)[0]:
        if I[coordinates[i][0],coordinates[i][1]] > 0.98:
            coordinates.pop(i)
        else:
            i+=1
    return np.asanyarray(coordinates)

# %%
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


