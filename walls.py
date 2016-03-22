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

        choice_list = [np.inf * np.ones_like(z)]

        u = np.zeros_like(z)        
        for w in self.walls:
            dz = w.get_distance(z)
            cond_list = [dz < self.R[type]]
            u += np.select(cond_list, choice_list)
        return u
