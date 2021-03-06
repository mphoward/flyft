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

    def get_bounds(self, sigma):
        min_bin = -1
        max_bin = self.system.Nbins
        if sigma is None or sigma is False:
            return min_bin, max_bin

        for w in self.walls:
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

        # ensure that the coefficients are set before we process
        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        upot = np.zeros_like(self.system.mesh)
        min_bin, max_bin = self.get_bounds(self.coeff.get(type, 'sigma'))
        if min_bin >= 0:
            upot[0:min_bin] = np.inf
        if max_bin < self.system.Nbins:
            upot[max_bin:] = np.inf

        bins = self.system.get_bin(z)
        return upot[bins]

class depletion(_wall_potential):
    def __init__(self, system, walls):
        """Depletion pair force between large hard spheres in a binary mixture

        Roth, Evans, and Dietrich, Phys. Rev. E 2000, 62, 5360 - 5377
        """
        _wall_potential.__init__(self, system, walls)
        if len(self.system.types) > 2:
            raise Exception('Depletion wall force should only be used in binary systems')

        self.coeff.require = ['sigma','rho','rcut']
        # flood the coefficient dictionary with defaults
        for t in self.system.types:
            self.coeff.set(t, sigma=self.system.sigma[t])

    def U(self, type, z):
        assert type in self.system.types
        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        upot = np.zeros_like(self.system.mesh)

        # pass on this type if it has no interaction
        rcut = self.coeff.get(type, 'rcut')
        if not rcut or not rcut > 0.0 or rcut is None:
            return upot

        sigma = self.coeff.get(type, 'sigma')
        rho = self.coeff.get(type, 'rho')
        if np.isclose(rho,0.0):
            return upot

        # trivial cases are taken care of, now do the hard work
        # exclude beyond the origin of the wall, which has a divergence
        min_bin, max_bin = self.get_bounds(sigma)
        if min_bin >= 0:
            upot[0:min_bin] = np.inf
        if max_bin < self.system.Nbins:
            upot[max_bin:] = np.inf

        Rs = 0.5 * rcut
        Rb = 0.5 * sigma
        # select the points that are inside the walls to do the calculation
        bins = self.system.get_bin(z)
        flags = np.logical_and(bins >= min_bin, bins < max_bin)
        for w in self.walls:
            cut_bin = self.system.get_bin(w.origin + w.normal * (rcut + Rb))
            if w.normal > 0:
                cut_flags = np.logical_and(flags, bins < cut_bin)
            else:
                cut_flags = np.logical_and(flags, bins >= cut_bin)

            if not np.any(cut_flags):
                continue

            h = w.normal * (z[cut_flags] - w.origin) - Rb

            upot[cut_flags] += -2.0 * rho * np.pi * (2.*Rs - h)*(Rs * (Rb + (1./3.)*Rs)-(h/2.)*(Rb + Rs/3.) - h**2/6.)
        return upot[bins]

class lj93(_wall_potential):
    def __init__(self, system, walls, shift=False):
        _wall_potential.__init__(self, system, walls)
        self.coeff.require = ['sigma','epsilon','rcut','shift']
        for t in self.system.types:
            self.coeff.set(t, shift=shift)

    def U(self, type, z):
        assert type in self.system.types
        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        upot = np.zeros_like(self.system.mesh)

        # pass on this type if it has no interaction
        rcut = self.coeff.get(type, 'rcut')
        if not rcut or not rcut > 0.0 or rcut is None:
            return upot

        sigma = self.coeff.get(type, 'sigma')
        eps = self.coeff.get(type, 'epsilon')
        if np.isclose(sigma, 0.0) or np.isclose(eps,0.0):
            return upot

        # trivial cases are taken care of, now do the hard work
        # exclude beyond the origin of the wall, which has a divergence
        min_bin, max_bin = self.get_bounds(0.0)
        if min_bin >= 0:
            upot[0:min_bin+2] = np.inf
        if max_bin < self.system.Nbins:
            upot[max_bin-1:] = np.inf

        # select the points that are inside the walls to do the calculation
        bins = self.system.get_bin(z)

        # precompute the LJ 9-3 constants
        A = (2./15.)*eps*sigma**9
#         A = eps*sigma**9
        B = eps*sigma**3
        U_cut = 0.0
        if self.coeff.get(type,'shift'):
            U_cut = A/rcut**9 - B/rcut**3

        flags = np.logical_and(bins > min_bin, bins < max_bin)
        for w in self.walls:
            cut_bin = self.system.get_bin(w.origin + w.normal * rcut)
            if w.normal > 0:
                cut_flags = np.logical_and(flags, bins < cut_bin)
            else:
                cut_flags = np.logical_and(flags, bins >= cut_bin)

            if not np.any(cut_flags):
                continue

            idz = 1. / (w.normal * (z[cut_flags] - w.origin))
            idz2 = idz*idz
            idz3 = idz2*idz
            idz9 = idz3*idz3*idz3

            upot[cut_flags] += A*idz9 - B*idz3 - U_cut
        return upot[bins]

class lj104(_wall_potential):
    def __init__(self, system, walls, shift=False):
        _wall_potential.__init__(self, system, walls)
        self.coeff.require = ['sigma','epsilon','rcut','shift']
        for t in self.system.types:
            self.coeff.set(t, shift=shift)

    def U(self, type, z):
        assert type in self.system.types
        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        upot = np.zeros_like(self.system.mesh)

        # pass on this type if it has no interaction
        rcut = self.coeff.get(type, 'rcut')
        if not rcut or not rcut > 0.0 or rcut is None:
            return upot

        sigma = self.coeff.get(type, 'sigma')
        eps = self.coeff.get(type, 'epsilon')
        if np.isclose(sigma, 0.0) or np.isclose(eps,0.0):
            return upot

        # trivial cases are taken care of, now do the hard work
        # exclude beyond the origin of the wall, which has a divergence
        min_bin, max_bin = self.get_bounds(0.0)
        if min_bin >= 0:
            upot[0:min_bin+2] = np.inf
        if max_bin < self.system.Nbins:
            upot[max_bin-1:] = np.inf

        # select the points that are inside the walls to do the calculation
        bins = self.system.get_bin(z)

        # precompute the LJ 10-4 constants
        A = (2./5.)*eps*sigma**10
        B = eps*sigma**4
        U_cut = 0.0
        if self.coeff.get(type,'shift'):
            U_cut = A/rcut**10 - B/rcut**4

        flags = np.logical_and(bins > min_bin, bins < max_bin)
        for w in self.walls:
            cut_bin = self.system.get_bin(w.origin + w.normal * rcut)
            if w.normal > 0:
                cut_flags = np.logical_and(flags, bins < cut_bin)
            else:
                cut_flags = np.logical_and(flags, bins >= cut_bin)

            if not np.any(cut_flags):
                continue

            idz = 1. / (w.normal * (z[cut_flags] - w.origin))
            idz2 = idz*idz
            idz4 = idz2*idz2
            idz8 = idz4*idz4

            upot[cut_flags] += A*(idz8*idz2) - B*idz4 - U_cut
        return upot[bins]

class harmonic(_wall_potential):
    def __init__(self, system, walls):
        _wall_potential.__init__(self, system, walls)
        self.coeff.require = ['k','offset','g','rcut']

    def U(self, type, z):
        assert type in self.system.types
        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        upot = np.zeros_like(z)

        # pass on this type if it has no interaction
        rcut = self.coeff.get(type, 'rcut')
        if not rcut or not rcut > 0.0 or rcut is None:
            return upot

        k = self.coeff.get(type, 'k')
        g = self.coeff.get(type, 'g')
        offset = self.coeff.get(type, 'offset')

        for w in self.walls:
            dz = w.normal * (z - (w.origin + offset))
            harm_flags = np.logical_and(dz >= 0., dz < rcut)
            upot[harm_flags] = 0.5 * k * dz[harm_flags] * dz[harm_flags]

            lin_flags = dz >= rcut
            upot[lin_flags] = 0.5 * k * rcut * rcut + g * (dz[lin_flags] - rcut)

        return upot
