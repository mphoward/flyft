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

        self.weights = (0,1,2,3,'v1','v2')

    def w(self, a, type, z):
        assert a in self.weights

        if a == 0:
            return (0.5 / self.R[type]) * np.ones_like(z)
        elif a == 1:
            return 0.5 * np.ones_like(z)
        elif a == 2:
            return 2.*np.pi*self.R[type] * np.ones_like(z)
        elif a == 3:
            return np.pi*(self.R[type]**2 - z*z)
        elif a == 'v1':
            # return as scalar because only nonzero entry is along ez
            return 0.5*z/self.R[type]
        elif a == 'v2':
            # return as scalar because only nonzero entry is along ez
            return 2.*np.pi*z

    def n(self, a, densities):
        assert a in self.weights

        n = np.zeros(self.system.Nbins)

        for t in self.system.types:
            z = np.arange(-self.R[t],self.R[t], self.system.dz)
            w = np.zeros(self.system.Nbins)
            # fill in and then roll to include boundaries correctly
            w[0:len(z)] = self.w(a,t,z)
            w = np.roll(w, -len(z)/2)

            n += self.system.dz * fft.convolve(densities[t], w, fft=True, periodic=True)
        return n

    def f1(self, n3):
        return -np.log(1.-n3)

    def df1(self,n3):
        return 1./(1.-n3)

    def f2(self, n3):
        return 1./(1.-n3)

    def df2(self, n3):
        return 1./(1.-n3)**2

    def f4(self, n3):
        return 1./(24.*np.pi*(1.-n3)**2)

    def df4(self, n3):
        return 1./(12.*np.pi*(1.-n3)**3)

    def dphi(self, densities):
        # precompute the weights
        n0 = self.n(0, densities)
        n1 = self.n(1, densities)
        n2 = self.n(2, densities)
        n3 = self.n(3, densities)
        nv1 = self.n('v1', densities)
        nv2 = self.n('v2', densities)

        if np.any(n3 > 1.0):
            raise Exception('n3 > 1.0, solution may be diverging!')

        # precompute the partials
        dphi_dn = {}
        dphi_dn[0] = self.f1(n3)
        dphi_dn[1] = self.f2(n3)*n2
        dphi_dn[2] = self.f2(n3)*n1 + 3.*self.f4(n3)*(n2*n2 - nv2*nv2)
        dphi_dn[3] = self.df1(n3)*n0 + self.df2(n3)*(n1*n2-nv1*nv2) + self.df4(n3)*(n2**3-3.*n2*nv2**2)
        dphi_dn['v1'] = -self.f2(n3)*nv2
        dphi_dn['v2'] = -self.f2(n3)*nv1 - 6.*self.f4(n3)*n2*nv2

        return dphi_dn

    def mu_ex(self, densities):
        dphi_dn = self.dphi(densities)

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

class whitebear(rosenfeld):
    def __init__(self, system, sigma={}):
        rosenfeld.__init__(self,system,sigma)

    def f4(self, n3):
        try:
            all_f4 = np.zeros(len(n3))
            n3_arr = np.array(n3)

            # flag entries of n3 that are big enough to use the actual formula
            flags = (n3_arr > whitebear._f4_threshold)

            # apply real formula to the "big" ones
            n3_big = n3_arr[flags]
            if len(n3_big) > 0:
                all_f4[flags] = (n3_big+(1.-n3_big)**2*np.log(1.-n3_big))/(36.*np.pi*n3_big**2*(1.-n3_big)**2)

            # use the taylor series for the "small" ones
            # this seems very accurate over the 1.e-2 range in Mathematica
            if len(n3_big) < len(n3_arr):
                n3_small = n3_arr[~flags]
                all_f4[~flags] = 1./(24.*np.pi) + 2.*n3_small/(27.*np.pi) + 5.*n3_small**2/(48.*np.pi)

            return all_f4

        except TypeError:
            # catch single values rather than arrays and just do scalar arithmetic
            if n3 > whitebear._f4_threshold:
                return (n3+(1.-n3)**2*np.log(1.-n3))/(36.*np.pi*n3**2*(1.-n3)**2)
            else:
                return 1./(24.*np.pi) + 2.*n3/(27.*np.pi) + 5.*n3**2/(48.*np.pi)

    def df4(self, n3):
        try:
            all_df4 = np.zeros(len(n3))
            n3_arr = np.array(n3)

            # flag entries of n3 that are big enough to use the actual formula
            flags = n3_arr > whitebear._f4_threshold

            # apply real formula to the "big" ones
            n3_big = n3_arr[flags]
            if len(n3_big) > 0:
                all_df4[flags] = -(2.-5.*n3_big+n3_big**2)/(36*np.pi*(1.-n3_big)**3*n3_big**2)-np.log(1.-n3_big)/(18.*np.pi*n3_big**3)

            # use the taylor series for the "small" ones
            # this seems very accurate over the 1.e-2 range in Mathematica
            if len(n3_big) < len(n3_arr):
                n3_small = n3_arr[~flags]
                all_df4[~flags] = 2./(27.*np.pi) + 5.*n3_small/(24.*np.pi) + 2.*n3_small**2/(5.*np.pi)

            return all_df4

        except TypeError:
            # catch single values rather than arrays and just do scalar arithmetic
            if n3 > whitebear._f4_threshold:
                return -(2.-5.*n3+n3**2)/(36*np.pi*(1.-n3)**3*n3**2)-np.log(1.-n3)/(18.*np.pi*n3**3)
            else:
                return 2./(27.*np.pi) + 5.*n3/(24.*np.pi) + 2.*n3**2/(5.*np.pi)

whitebear._f4_threshold = 1.e-2
