import numpy as np
import random
# %% 
def _base ( settings, params, data):
    '''
    :param settings: general settings used
    :param params: general parameters used
    :param data: data loaded from file
    :return:
    '''
    n_images = len(data.images)
    for i in range(n_images): # the search is performed for each image individually (for the first time point)
        img = data.images[i]
        img -= np.mean(img) # subtract the mean of the image to start

        templates = generate_templates(params,img)

# %%



def generate_templates ( params,img ):
    '''
    Templates are generated based on Parameter-Free Binarization and Skeletonization of Fiber Networks from Confocal Image Stacks
    The searching axis will always be 0-1 so the input image will have to be transposed for the axis to change from one
    iteration to the next.
    '''

    random.seed(params.seed)
    sample_idx = random.sample(list(range(img.size)),
                               params.num_nodes_generate)  # draw indices that we will later use to search in

    shape = np.shape(img)

    template_size_x = int(shape[0] * params.template_size_x) # initial template size
    template_size_y = int(shape[1] * params.template_size_y) # initial template size
    x_y_template = extract_volumes(params,img,sample_idx,template_size_x,template_size_y)

    ''' permute shape to fit new search direction '''


def extract_volumes ( params , img, sample_idx,template_size_0,template_size_1 ):
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

    slicing_indices = []

    for i in range(params.num_nodes_generate):
        slicing_indices.append(np.unravel_index(sample_idx[i],img.shape))
    # %%


    for i,indices in enumerate(slicing_indices):
        min0 = np.max((0,slicing_indices[i][0]-template_size_0))
        max0 = np.min((np.shape(img)[0], slicing_indices[i][0] + template_size_0))
        min0 = np.max((0, slicing_indices[i][1] - template_size_1))
        max0 = np.min((np.shape(img)[1], slicing_indices[i][1] + template_size_1))

        Area = img[min0:max0,min1:max1,slicing_indices[i][2]]/img[slicing_indices[0]]


        # if i == 0:
        #     sample = img[min_x:max_x,min_y:max_y,min_z:max_z]

        flag = 1
    flag = 1