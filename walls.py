import numpy as np

class _wall_potential(object):
    def __init__(self, system, walls):
        self.system = system

        try:
            Nwalls = len(walls)
            self.walls = walls
        except TypeError:
            self.walls = [walls]

class hard(_wall_potential):
    def __init__(self, system, walls, sigma={}):
        _wall_potential.__init__(self, system, walls)

        self.R = {}
        for t in self.system.types:
            if t in sigma:
                self.R[t] = 0.5*sigma[t]
            elif self.system.sigma[t] is not None:
                self.R[t] = 0.5*self.system.sigma[t]
            else:
                raise Exception('sigma must be set for all species to use HS')

    def U(self, type, z):
        assert type in self.system.types

        min_bin = 0
        max_bin = self.system.Nbins - 1
        for w in self.walls:
            if w.normal > 0:
                min_bin = max(self.system.get_bin(w.origin + self.R[type]), min_bin)
            else:
                max_bin = min(self.system.get_bin(w.origin - self.R[type]), max_bin)

        upot = np.zeros_like(self.system.mesh)
        upot[0:min_bin] = np.inf
        upot[(max_bin+1):] = np.inf

        bins = self.system.get_bin(np.array(z))
        return upot[bins]
