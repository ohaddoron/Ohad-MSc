import javabridge
import bioformats
import os
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import pickle
from tqdm import tqdm
from PIL import Image

class loaded_data:
    def __init__(self):
        self.images = ()
        self.names = ()

class metadata:
    def __init__(self,omexmlstr):
        o = bioformats.OMEXML(omexmlstr)
        pixels = o.image().Pixels
        self.num_pos = o.get_image_count()
        self.num_channels = pixels.get_channel_count()
        self.num_planes = pixels.SizeZ
        self.num_time_points = pixels.SizeT
        self.num_X = pixels.SizeX
        self.num_Y = pixels.SizeY

def load_single_file ( settings,rdr,mdata ):
    '''
    loads a single data file
    Inputs: settings - a structure containing all required settings for run. Created using the load_settings_params module
    Outputs: 5D array of X,Y,Z,T,P
    '''
    '''
    init: 
    '''
    num_cores = multiprocessing.cpu_count()


    for pos in tqdm(range(mdata.num_pos)):
        img = load_position(rdr,mdata,pos)
        if 'I' not in locals():
            I = img
        else:
            try:
                I = np.concatenate((I,img[...,np.newaxis]),axis = 4)
            except:
                I = np.concatenate((I[...,np.newaxis],img[...,np.newaxis]), axis =4)
    return I



def load_position(rdr,mdata,pos):
    '''
    loads all planes in all time points for a single position.
    '''

    for TP in range(mdata.num_time_points):
        for plane in range(mdata.num_planes):
            I = rdr.read(z=plane, t=TP, series=pos)
            if len(np.shape(I)) > 2:
                I = rdr.read(z=plane, t=TP, series=pos)[:,:,0]
            if plane == 0:
                tmp_img = I
            else:
                try:
                    tmp_img = np.concatenate(
                        (tmp_img, I[..., np.newaxis]),axis=2)
                except:
                    tmp_img = np.concatenate((tmp_img[...,np.newaxis], I[...,np.newaxis]),axis=2)
        if not 'img' in locals():
            img = tmp_img
        else:
            try:
                img = np.concatenate((img,tmp_img[...,np.newaxis]),axis=3)
            except:
                img = np.concatenate((img[...,np.newaxis],tmp_img[...,np.newaxis]),axis=3)
    return img

def convert_images(settings,params):
    javabridge.start_vm(class_path=bioformats.JARS)
    I = ()
    # currently assuming only 1 channel of interest 12/11/17 - OD
    for file in os.listdir(settings.path2data):
        if file.endswith(settings.file_format):
            fname = os.path.join(settings.path2data, file)
            rdr = bioformats.ImageReader(path=fname)
            mdata = metadata(bioformats.get_omexml_metadata(fname))

            I = load_single_file(settings,rdr,mdata)
            new_name = os.path.join(settings.path2data,file.partition('.')[0])
            dumpImages2File(new_name,I)



def dumpImages2File(fname,I):
    filename = fname +'.pkl'
    with open(filename,'w') as f:
        pickle.dump(I,f)

def loadImagesFromFile ( settings ):
    '''
    the data return is a tuple with all image data loaded into in.
    the dimensions are sorted in the following order:
    1st - X
    2nd - Y
    3rd - Z
    4th - T
    5th - P
    '''
    for file in os.listdir(settings.path2data):
        fname = os.path.join(settings.path2data,file)
        if file.endswith('pkl'):
            with open(fname,'r') as f:
                if not 'data' in locals():
                    data = loaded_data()
                data.images += (np.load(f),)
                data.names += (file,)

    return data

