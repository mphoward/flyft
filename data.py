import numpy as np

class density(object):
    """Density object
    """
    def __init__(self, bulk):
        self.bulk = bulk
        self.local = None

    def discretize(self, zmin, zmax, dz):
        self.local = np.arange(zmin,zmax+dz,dz)