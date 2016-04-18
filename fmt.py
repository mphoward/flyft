import numpy as np
import fft
import data

class rosenfeld(object):
    def __init__(self, system, sigma={}):
        self.system = system

        self.coeff = data.coeff(self.system, require='sigma')
        # flood the coefficient dictionary with defaults
        for t in self.system.types:
            self.coeff.set(t, sigma=self.system.sigma[t])

        self.weights = (0,1,2,3,'v1','v2')

    def w(self, a, type, z):
        assert a in self.weights

        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        sigma = self.coeff.get(type, 'sigma')
        if self._is_ideal(sigma):
            return np.zeros_like(z)

        R = 0.5*sigma
        if a == 0:
            return (0.5 / R) * np.ones_like(z)
        elif a == 1:
            return 0.5 * np.ones_like(z)
        elif a == 2:
            return 2.*np.pi*R * np.ones_like(z)
        elif a == 3:
            return np.pi*(R**2 - z*z)
        elif a == 'v1':
            # return as scalar because only nonzero entry is along ez
            return 0.5*z/R
        elif a == 'v2':
            # return as scalar because only nonzero entry is along ez
            return 2.*np.pi*z

    def wk(self, a, type, k):
        assert a in self.weights
        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        sigma = self.coeff.get(type, 'sigma')
        wk = np.zeros_like(k, dtype=np.complex_)
        if self._is_ideal(sigma):
            return wk

        R = 0.5*sigma
        omega = 2.0 * np.pi * k
        flags = ~np.isclose(omega, 0.0)
        if a == 0:
            wk[flags] = np.sin(omega[flags] * R) / (omega[flags] * R)
            wk[~flags] = 1.0
        elif a == 1:
            wk[flags] = np.sin(omega[flags] * R)/ omega[flags]
            wk[~flags] = R
        elif a == 2:
            wk[flags] = 2.0 * R * np.sin(omega[flags] * R) / k[flags]
            wk[~flags] = 4.0 * np.pi * R**2
        elif a == 3:
            wk[flags] = (2.0/k[flags]) * (np.sin(omega[flags] * R) - omega[flags] * R * np.cos(omega[flags] * R))/omega[flags]**2
            wk[~flags] = (4.0*np.pi*R**3)/3.0
        elif a == 'v1':
            wk[flags] = -1.j * (np.sin(omega[flags] * R) - omega[flags] * R * np.cos(omega[flags] * R))/(R * omega[flags]**2)
            wk[~flags] = 0.0
        elif a == 'v2':
            wk[flags] = -1.j * (4.0 * np.pi) * (np.sin(omega[flags] * R) - omega[flags] * R * np.cos(omega[flags] * R))/omega[flags]**2
            wk[~flags] = 0.0
        return wk

    def n(self, a, densities, bulk=False):
        assert a in self.weights

        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        if bulk:
            n = np.zeros_like(densities[self.system.types[0]])
            for t in self.system.types:
                sigma = self.coeff.get(t,'sigma')
                if self._is_ideal(sigma):
                    continue

                if a == 0:
                    n += densities[t]
                elif a == 1:
                    n += densities[t] * 0.5*sigma
                elif a == 2:
                    n += 4.0 * np.pi * (0.5*sigma)**2 * densities[t]
                elif a == 3:
                    n += (4.0 * np.pi / 3.0) * (0.5*sigma)**3 * densities[t]
                elif a == 'v1' or a == 'v2':
                    n += 0.0
        else:
            n = np.zeros_like(self.system.mesh)
            k_mesh = np.fft.fftfreq(len(self.system.mesh), self.system.dz)

            for t in self.system.types:
                sigma = self.coeff.get(t,'sigma')
                if self._is_ideal(sigma):
                    continue

                rho = np.array(densities[t])
                n += np.real(np.fft.ifft(np.fft.fft(rho) * self.wk(a,t,k_mesh)))

        return n

    def _is_ideal(self,sigma):
        """Check if particle diameter signals ideal"""
        return sigma is None or not sigma > 0.0 or sigma is False

    def f1(self, n3):
        return -np.log(1.-n3)

    def df1(self,n3):
        return 1./(1.-n3)

    def ddf1(self, n3):
        return 1./(1.-n3)**2

    def f2(self, n3):
        return 1./(1.-n3)

    def df2(self, n3):
        return 1./(1.-n3)**2

    def ddf2(self, n3):
        return 2./(1.-n3)**2

    def f4(self, n3):
        return 1./(24.*np.pi*(1.-n3)**2)

    def df4(self, n3):
        return 1./(12.*np.pi*(1.-n3)**3)

    def ddf4(self, n3):
        return 1./(4.*np.pi*(1.-n3)**4)

    def phi(self, densities):
        # precompute the weights
        n0 = self.n(0, densities)
        n1 = self.n(1, densities)
        n2 = self.n(2, densities)
        n3 = self.n(3, densities)
        nv1 = self.n('v1', densities)
        nv2 = self.n('v2', densities)

        if np.any(n3 > 1.0):
            raise Exception('n3 > 1.0, solution may be diverging!')

        return self.f1(n3)*n0 + self.f2(n3)*(n1*n2 - nv1*nv2) + self.f4(n3)*(n2**3 - 3.*n2*nv2**2)

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

    def dphi2(self,densities):
        # precompute the weights
        weight_types = (0,1,2,3,'v1','v2')

        n = {}
        dphi2_dn = {}
        for a in weight_types:
            n[a] = self.n(a, densities)
            dphi2_dn[a] = {}
            for b in weight_types:
                dphi2_dn[a][b] = np.zeros_like(n[a])

        if np.any(n[3] > 1.0):
            raise Exception('n3 > 1.0, solution may be diverging!')

        # n0 derivatives
        dphi2_dn[3][0] = dphi2_dn[0][3] = self.df1(n[3])
        # n1 derivatives
        dphi2_dn[2][1] = dphi2_dn[1][2] = self.f2(n[3])
        dphi2_dn[3][1] = dphi2_dn[1][3] = self.df2(n[3]) * n[2]
        # n2 derivatives
        dphi2_dn[2][2] = 6.0 * self.f4(n[3]) * n[2]
        dphi2_dn[3][2] = dphi2_dn[2][3] = self.df2(n[3]) * n[1] + 3.0 * self.df4(n[3]) * (n[2]**2 - n['v2']**2)
        dphi2_dn['v2'][2] = dphi2_dn[2]['v2'] = -6.0 * self.f4(n[3]) * n['v2']
        # n3 derivatives
        dphi2_dn['v1'][3] = dphi2_dn[3]['v1'] = -self.df2(n[3]) * n['v2']
        dphi2_dn['v2'][3] = dphi2_dn[3]['v2'] = -self.df2(n[3]) * n['v1'] - 6.0 * self.df4(n[3]) * n[2] * n['v2']
        dphi2_dn[3][3] = self.ddf1(n[3]) * n[0] + self.ddf2(n[1] * n[2] - n['v1'] * n['v2']) + self.ddf4(n[3]) * (n[2]**3 - 3.0 * n[2] * n['v2']**2)
        # nv1 derivatives
        dphi2_dn['v2']['v1'] = dphi2_dn['v1']['v2'] = -self.f2(n[3])
        # nv2 derivatives
        dphi2_dn['v2']['v2'] = -6.0 * self.f4(n[3]) * n[2]

        return dphi2_dn


    def F_ex(self, densities):
        phi = self.phi(densities)
        return self.system.dz * np.sum(phi)

    def mu_ex(self, densities):
        dphi_dn = self.dphi(densities)

        if not self.coeff.verify():
            raise Exception('not all parameters are set!')

        # initialize for summation
        mu_ex = {}

        for t in self.system.types:
            mu_ex[t] = np.zeros_like(self.system.mesh)
            k_mesh = np.fft.fftfreq(len(self.system.mesh), self.system.dz)

            sigma = self.coeff.get(t, 'sigma')
            if self._is_ideal(sigma):
                continue

            z = np.arange(-0.5*sigma,0.5*sigma+self.system.dz, self.system.dz)

            for a in (0,1,2,3,'v1','v2'):
                sign = 1.0
                if a == 'v1' or a == 'v2':
                    sign = -1.0
                mu_ex[t] += np.real(np.fft.ifft(np.fft.fft(dphi_dn[a]) * sign * self.wk(a,t,k_mesh)))

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
