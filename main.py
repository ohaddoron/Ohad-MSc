import load_data
import load_settings_params

if __name__ == '__main__':
    settings = load_settings_params.settings()
    params = load_settings_params.params()
    if settings.load_operation == 'convert':
        '''
        convert czi files into pkl files of images
        '''
        load_data.convert_images(settings,params)
    elif settings.load_operation == 'load':
        '''
        data is organized as follows:
        first layer - positions 
        second layer - time points
        third layer - X
        fourth layer - Y
        fifth layer - Z
        '''
        data = load_data.loadImagesFromFile(settings)
    a = 1