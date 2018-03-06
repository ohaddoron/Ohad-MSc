import os
# %% 
class settings:
    def __init__(self):

        self.path2data = os.path.join('..', 'data')
        self.path2results = os.path.join('..','results')
        self.path2figures = os.path.join('..','figures')
        self.file_format = 'czi'
        self.save_format = 'npy'
        self.fraction_cores = 3.0/8.0
        self.parallel = True
        self.load_operation = 'load' # load or convert
        self.visualize_node_detection = False

# %%
class params:
    def __init__(self):
        self.seed = 1
        self.node_size = 15 # number of pixels to extract in the x and y directions
        self.thresh = 0.6 # intensity threshold to be concidered as a cross section (node)
        self.num_nodes_generate = int(1e5)
        self.template_size_x = 0.025 # fraction of the original image size. fraction may vary for different densities
        self.template_size_y = 0.025 # fraction of the original image size. fraction may vary for different densities
        self.template_size_z = 0.1 # fraction of the original image size. fraction may vary for different densities
        self.RG_thresh = 10


# %% 
def load():
    return settings(),params()
