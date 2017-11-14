import os

class settings:
    def __init__(self):

        self.path2data = os.path.join('..', 'data')
        self.path2results = os.path.join('..','results')
        self.path2figures = os.path.join('..','figures')
        self.file_format = 'czi'
        self.fraction_cores = 3.0/8.0
        self.parallel = True
        self.load_operation = 'load' # load or convert

class params:
    def __init__(self):
        ''''''

