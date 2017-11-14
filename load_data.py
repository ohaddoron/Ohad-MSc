import javabridge
import bioformats
import os
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import pickle
from tqdm import tqdm

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
    if settings.parallel:
        # Parallel is not working for an unknown reason. - 11/14/17 OD
        I = Parallel(n_jobs=int(np.floor((settings.fraction_cores) * num_cores)))(
            delayed(load_position)(rdr,mdata,pos) for pos in range(mdata.num_pos))
    else:
        I = ()
        for pos in tqdm(range(mdata.num_pos)):
            I += (load_position(rdr,mdata,pos),)
    return I



def load_position(rdr,mdata,pos):
    '''
    loads all planes in all time points for a single position.

    '''
    img = ()
    for TP in tqdm(range(mdata.num_time_points)):
        for plane in range(mdata.num_planes):
            if plane == 0:
                tmp_img = rdr.read(z=plane, t=TP, series=pos)
            else:
                tmp_img = np.dstack((tmp_img, rdr.read(z=plane, t=TP, series=pos)))
        img += (tmp_img,)
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
    with open(fname + '.pkl','w') as f:
        pickle.dump(I,f)

def loadImagesFromFile ( settings ):

    for file in os.listdir(settings.path2data):
        fname = os.path.join(settings.path2data,file)
        if file.endswith('pkl'):
            with open(fname,'r') as f:
                if not 'data' in locals():
                    data = loaded_data()
                data.images += (pickle.load(f),)
                data.names += (file,)

    return data

