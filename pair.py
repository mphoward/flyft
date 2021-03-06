import numpy as np
import data

class _wca(object):
    def __init__(self, system):
        self.system = system
        self.coeff = data.pair_coeff(self.system)

    def get_rmin(self, i, j):
        raise Exception('programming error: potential minimum evaluator must be defined!')

    def energy(self, i, j):
        raise Exception('programming error: potential evaluator must be defined!')

    def U(self, i, j, r):
        if not self.coeff.verify():
            raise Exception('all required coefficients are not set!')

        upot = np.zeros_like(r)
        rs = np.fabs(np.array(r))

        rmin = self.get_rmin(i, j)
        flag_eval = rs >= rmin
        upot[~flag_eval] = self.energy(i, j, rmin)
        upot[flag_eval] = self.energy(i, j, rs[flag_eval])

        return upot

class lj(_wca):
    def __init__(self, system):
        _wca.__init__(self, system)
        self.coeff.require = ['sigma','epsilon','rcut','shift']

        # the default sigma is the hard sphere contact (i.e., arithmetic mixing)
        for i in self.system.types:
            for j in self.system.types:
                if self.system.sigma[i] is not None and self.system.sigma[j] is not None:
                    sigma_ij = 0.5*(self.system.sigma[i] + self.system.sigma[j])
                    self.coeff.set(i, j, sigma=sigma_ij)

    def get_rmin(self, i, j):
        return np.power(2.0,1.0/6.0) * self.coeff.get(i, j, 'sigma')

    def energy(self, i, j, r):
        U = np.zeros_like(r)
        rs = np.array(r)

        # pass on this type if it has no interaction
        rcut = self.coeff.get(i, j, 'rcut')
        if not rcut or not rcut > 0.0 or rcut is None:
            return U

        # potential is zero if either sigma or epsilon is close to zero
        sigma = self.coeff.get(i, j, 'sigma')
        eps = self.coeff.get(i, j, 'epsilon')
        if np.isclose(sigma, 0.0) or np.isclose(eps,0.0):
            return U

        shift = self.coeff.get(i, j,'shift')
        U_cut = 0.0
        if shift:
            U_cut = 4.*eps*((sigma/rcut)**12 - (sigma/rcut)**6)

        # only perform calculations for points inside the rcut
        flags = (np.fabs(rs) <= rcut)
        ri = sigma/rs[flags]
        r2i = ri*ri
        r4i = r2i*r2i
        r6i = r2i*r4i
        r12i = r6i*r6i
        U[flags] = 4.*eps*(r12i - r6i) - U_cut

        return U

class depletion(object):
    def __init__(self, system):
        """Depletion pair force between large hard spheres in a binary mixture

        Roth, Evans, and Dietrich, Phys. Rev. E 2000, 62, 5360 - 5377
        """
        self.system = system
        if len(self.system.types) > 2:
            raise Exception('Depletion pair force should only be used in binary systems')

        self.coeff = data.pair_coeff(self.system, require=['sigma', 'rho', 'rcut'])

    def U(self, i, j, r):
        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        upot = np.zeros_like(r)
        # pass on this type if it has no interaction
        rcut = self.coeff.get(i, j, 'rcut')
        if not rcut or not rcut > 0.0 or rcut is None:
            return upot

        sigma = self.coeff.get(i, j, 'sigma')
        rho = self.coeff.get(i, j, 'rho')
        if np.isclose(rho,0.0):
            return upot

        rs = np.fabs(np.array(r))

        Rs = 0.5 * rcut
        Rb = 0.5 * sigma
        flag_eval = np.logical_and(rs >= Rb, rs <= rcut)
        h = r[flag_eval] - Rb
        upot[flag_eval] = -rho * np.pi * (2.*Rs - h)*(Rs * (Rb + (2./3.)*Rs)-(h/2.)*(Rb + Rs/3.) - h**2/12.)
        return upot
