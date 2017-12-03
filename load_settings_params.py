import os

class settings:
    def __init__(self):

        self.path2data = os.path.join('..', 'data')
        self.path2results = os.path.join('..','results')
        self.path2figures = os.path.join('..','figures')
        self.file_format = 'czi'
        self.save_format = 'npy'
        self.fraction_cores = 3.0/8.0
        self.parallel = True
        self.load_operation = 'convert' # load or convert
        self.visualize_node_detection = False

class params:
    def __init__(self):
        self.max_filter_kernel_size = 15
        self.med_filter_kernel_size = 5
        self.peak_detector_size = 15
        self.dist_threshold = 5
        self.node_size = 15


