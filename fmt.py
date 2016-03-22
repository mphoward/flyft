import numpy as np
import fft

class rosenfeld(object):
    def __init__(self, system, sigma={}):
        self.system = system

        self.R = {}
        for t in self.system.types:
            if t in sigma:
                self.R[t] = 0.5*sigma[t]
            elif self.system.sigma[t] is not None:
                self.R[t] = 0.5*self.system.sigma[t]
            else:
                raise Exception('sigma must be set for all species to use HS')

    def w0(self, type, z):
        if len(z) > 1:
            return (0.5 / self.R[type]) * np.ones_like(z)
        else:
            return (0.5 / self.R[type])

    def w1(self, type, z):
        if len(z) > 1:
            return 0.5 * np.ones_like(z)
        else:
            return 0.5

    def w2(self, type, z):
        if len(z) > 1:
            return 2.*np.pi*self.R[type] * np.ones_like(z)
        else:
            return 2.*np.pi*self.R[type]

    def w3(self, type, z):
        return np.pi*(self.R[type]**2 - z*z)

    def wv1(self, type, z):
        # return as scalar because only nonzero entry is along ez
        return 0.5*z/self.R[type]

    def wv2(self, type, z):
        # return as scalar because only nonzero entry is along ez
        return 2.*np.pi*z

    def w(self, a, type, z):
        assert a in (0,1,2,3,'v1','v2')
        if a == 0:
            return self.w0(type,z)
        elif a == 1:
            return self.w1(type,z)
        elif a == 2:
            return self.w2(type,z)
        elif a == 3:
            return self.w3(type,z)
        elif a == 'v1':
            return self.wv1(type,z)
        elif a == 'v2':
            return self.wv2(type,z)

    def n(self, a, densities):
        assert a in (0,1,2,3,'v1','v2')

        n = np.zeros(self.system.Nbins)

        for t in self.system.types:
            z = np.arange(-self.R[t],self.R[t], self.system.dz)
            w = np.zeros(self.system.Nbins)
            # fill in and then roll to include boundaries correctly
            w[0:len(z)] = self.w(a,t,z)
            w = np.roll(w, -len(z)/2)

            n += self.system.dz * fft.convolve(densities[t], w, fft=True, periodic=True)
        return n

    def mu_ex(self, densities):
        # precompute the weights
        n0 = self.n(0, densities)
        n1 = self.n(1, densities)
        n2 = self.n(2, densities)
        n3 = self.n(3, densities)
        nv1 = self.n('v1', densities)
        nv2 = self.n('v2', densities)

        # precompute the partials
        dphi_dn = {}
        dphi_dn[0] = -np.log(1.-n3)
        dphi_dn[1] = n2/(1.-n3)
        dphi_dn[2] = n1/(1.-n3) + (n2 - nv2*nv2)/(8.*np.pi*(1.-n3)**2)
        dphi_dn[3] = n0/(1.-n3) + (n1*n2 - nv1*nv2 + n2**3/(12.*np.pi))/(1.-n3)**2 - n2*nv2**2/(4.*np.pi*(1.-n3)**3)
        dphi_dn['v1'] = -nv2/(1.-n3)
        dphi_dn['v2'] = -nv1/(1.-n3) - n2*nv2/(4.*np.pi*(1.-n3)**2)

        # initialize for summation
        mu_ex = {}

        for t in self.system.types:
            z = np.arange(-self.R[t],self.R[t], self.system.dz)
            mu_ex[t] = np.zeros(self.system.Nbins)

            for a in (0,1,2,3,'v1','v2'):
                w = np.zeros(self.system.Nbins)
                # fill in and then roll to include boundaries correctly
                w[0:len(z)] = self.w(a,t,z)
                w = np.roll(w, -len(z)/2)
                sign = 1.0
                if a == 'v1' or a == 'v2':
                    sign = -1.0

                mu_ex[t] += self.system.dz * fft.convolve(dphi_dn[a], sign*w, fft=True, periodic=True)
        return mu_ex