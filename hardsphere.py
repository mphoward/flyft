import numpy as np

class fmt(object):
    def __init__(self, N, density, R):
        assert isint(N) and N >= 1
        self.N = N

        if self.N == 1:
            self.density = [density]
            self.R = [R]
        else:
            self.density = density
            self.R = R
        self.R = np.array(self.R)

        assert len(self.density) == N
        assert len(self.R) == N

        # precache some of the weights that are constants
        self._w0 = 0.5 / self.R
        self._w1 = 0.5
        self._w2 = 2.*np.pi*self.R
        self.Rsq = self.R*self.R

    def w0(self, i, z):
        assert i >= 0 and i < N
        return self._w0[i]

    def w1(self, i, z):
        assert i >= 0 and i < N
        return self._w1[i]

    def w2(self, i, z):
        assert i >= 0 and i < N
        return self._w2[i]

    def w3(self, i, z):
        assert i >= 0 and i < N
        return np.pi*(self.Rsq[i] - z*z)

    def wv1(self, i, z):
        assert i >= 0 and i < N
        return self.wv2(i,z) / (4.*np.pi*self.R[i])

    def wv2(self, i, z):
        assert i >= 0 and i < N
        return np.array([0., 0., 2.*np.pi*z])

    def n(self, a):
        assert a in (0,1,2,3,'v1','v2')
