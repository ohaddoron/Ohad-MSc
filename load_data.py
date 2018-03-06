import javabridge
import bioformats
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm



#%%
class converter:
    #%%
    def __init__(self,settings,params):
        self.convert_images(settings,params)
    #%%
    def convert_images(self,settings,params):
        javabridge.start_vm(class_path=bioformats.JARS)
        # currently assuming only 1 channel of interest 12/11/17 - OD
        for file in os.listdir(settings.path2data):
            if file.endswith(settings.file_format):
                fname = os.path.join(settings.path2data, file)
                self.extract(settings,fname)
    #%%            
    def get_metadata(self,omexmlstr):
        o = bioformats.OMEXML(omexmlstr)
        pixels = o.image().Pixels
        metadata = {}
        metadata['num_pos'] = o.get_image_count() # number of positions in file
        metadata['num_channels'] = pixels.get_channel_count() # number of channels in file
        metadata['num_planes'] = pixels.SizeZ # number of different planes (size of Z stack)
        metadata['num_time_points'] = pixels.SizeT # number of time points taken
        metadata['num_X'] = pixels.SizeX # number of pixels in the X dimension
        metadata['num_Y'] = pixels.SizeY # number of pixels in the Y dimension
        return metadata
    
    
    #%% 
    def extract ( self, settings,fname):
        '''
        loads a single data file
        Inputs: settings - a structure containing all required settings for run. Created using the load_settings_params module
        Outputs: 5D array of X,Y,Z,T,P
        '''

        # init: 
        metadata = self.get_metadata(bioformats.get_omexml_metadata(fname))
        rdr = bioformats.ImageReader(path=fname)
        
        # num_cores = int(np.floor(multiprocessing.cpu_count()/2))
        for pos in tqdm(range(metadata['num_pos'])): # cycle through all positions, load position from meta file and dump it into file for later use
            tmp = self.read_time_points(rdr,metadata,pos)
            if 'I' not in locals():
                I = tmp
            else:
                I = np.stack((I,tmp),axis=4)      
        np.save(os.path.join(settings.path2data,fname.replace('czi','npy')),I)
    #%%    
    def read_time_points ( self,rdr, metadata, pos ):
        for t in range(metadata['num_time_points']):
            tmp = self.read_planes(rdr,metadata,t,pos)
            if 'I' not in locals():
                I = tmp
            else:
                if t > 1:
                    I = np.concatenate((I,tmp[...,np.newaxis]),axis=3)
                elif t == 1:
                    I = np.stack((I,tmp),axis=3)
        I[np.where(I < 0)] -= 2 * I[np.where(I<0)]
        return I
    
    #%% 
    def read_planes(self,rdr,metadata,t,pos):
        for z in range(metadata['num_planes']):
            tmp = rdr.read(z=z,t=t,series=pos)[:,:,0] # this assumes there exists only one channel
            if 'I' not in locals():
                I = tmp
            else:
                I = np.dstack((I,tmp))
        return I
#%%
class loader:
    #%%
    def load_images(self,settings):
        data = {'names':[],'images':[]}
        for file in os.listdir(settings.path2data):
            if file.endswith('.npy'):
                data['names'].append(file)
                data['images'].append(np.load(os.path.join(settings.path2data,file)))
        return data
    

