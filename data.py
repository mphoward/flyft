import numpy as np

class system(object):
    """Simulation system
    Parameters
    ----------
    L : float
        Length of the simulation box

    dz : float
        Mesh size (must evenly divide L)

    """
    def __init__(self, L, dz):
        assert int(L / dz) * dz == L

        self.L = L
        self.dz = dz
        self.Nbins = int(self.L/self.dz)
        self.mesh = -0.5*self.L + np.arange(0,self.Nbins)*self.dz + 0.5*self.dz

        # system walls
        self.walls = []

        # fluid data stored per type
        self.types = []
        self.density = {}
        self.bulk = {}
        self.sigma = {}

    def add_type(self, name, bulk, sigma=None):
        assert not name in self.types

        self.types += [name]

        self.bulk[name] = bulk * np.ones(self.Nbins)

        self.density[name] = np.zeros(self.Nbins)

        self.sigma[name] = sigma

    def get_z(self, bin):
        assert bin >= 0 and bin < self.Nbins

        return -0.5*self.L + bin*self.dz

    def get_bin(self, z):
        try:
            bin = ((np.array(z)+0.5*self.L)/self.dz)
            return bin.astype(int)
        except TypeError:
            return int(bin)

class wall(object):
    def __init__(self, system, origin, normal):
        self.system = system
        self.origin = origin

        if normal > 0:
            self.normal = 1.
        elif normal < 0:
            self.normal = -1.
        else:
            raise Exception('wall normal must point in a direction!')

        self.potential = None

    def get_distance(self, z):
        return self.normal * (z - self.origin)
