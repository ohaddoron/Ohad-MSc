from load_data import converter,loader
import load_settings_params
import numpy as np
from matplotlib import pyplot as plt
import feature_extraction

if __name__ == '__main__':
    settings,params = load_settings_params.load()
    
    if settings.load_operation == 'convert':
        # %% convert
        '''
        convert czi files into pkl files of images
        '''
        converter(settings,params)

    elif settings.load_operation == 'load':
         # %% load
        '''
        data is organized as follows:
        first layer - positions 
        second layer - time points
        third layer - X
        fourth layer - Y
        fifth layer - Z
        '''

        data = loader().load_images(settings)
        feature_extraction._base(settings,params,data)


