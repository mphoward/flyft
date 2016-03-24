import numpy as np
import data

class _wall_potential(object):
    def __init__(self, system, walls):
        self.system = system

        try:
            Nwalls = len(walls)
            self.walls = walls
        except TypeError:
            self.walls = [walls]

        self.coeff = data.coeff(self.system)

    def get_bounds(self, type):
        # ensure that the coefficients are set before we process
        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        min_bin = -1
        max_bin = self.system.Nbins
        for w in self.walls:
            sigma = self.coeff.get(type,'sigma')
            if sigma is None or not sigma > 0.0 or sigma is False:
                continue

            if w.normal > 0:
                min_bin = max(self.system.get_bin(w.origin + 0.5*sigma), min_bin)
            else:
                max_bin = min(self.system.get_bin(w.origin - 0.5*sigma), max_bin)

        return min_bin, max_bin

class hard(_wall_potential):
    def __init__(self, system, walls):
        _wall_potential.__init__(self, system, walls)

        self.coeff.require = ['sigma']
        # flood the coefficient dictionary with defaults
        for t in self.system.types:
            self.coeff.set(t, sigma=self.system.sigma[t])

    def U(self, type, z):
        assert type in self.system.types

        min_bin, max_bin = self.get_bounds(type)

        upot = np.zeros_like(self.system.mesh)
        if min_bin >= 0:
            upot[0:min_bin] = np.inf
        if max_bin < self.system.Nbins:
            upot[max_bin:] = np.inf

        bins = self.system.get_bin(np.array(z))
        return upot[bins]
