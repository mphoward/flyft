import numpy as np

# python 2/3 compatibility layer for string testing
try:
    basestring
except NameError:
    basestring = str

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

class coeff(object):
    """Coefficient dictionary

    Parameters
    ----------
    system : data.system
        The system to attach to

    require : array like, optional
        List of parameter fields to require in order to validate the coefficient
        dictionary

    Examples
    --------
    my_coeff = coeff(system)
    my_coeff = coeff(system, require=['foo','bar'])

    """
    def __init__(self, system, require=None):
        self.system = system

        if isinstance(require,basestring):
            self.require = [require]
        else:
            self.require = require

        self._params = {}

        # flag to signal revalidation if None
        # otherwise just return the cached value
        self.__valid = None

    def set(self, type, **coeffs):
        """Set a coefficient entry

        Parameters
        ----------
        type : string or array_like
            Particle type(s) to enter coefficient

        coeffs : keyword arguments
            Coefficients to enter as keyword arguments

        Examples
        --------
        my_coeff.set('A', sigma=1.0, epsilon=2.0)
        my_coeff.set(['B','C'], sigma=3.0)
        """
        if isinstance(type, basestring):
            type = [type]

        for t in type:
            # force the type into the parameter dict
            if not t in self._params:
                self._params[t] = {}

            for key, val in coeffs.iteritems():
                self._params[t][key] = val

        self.__valid = None

    def get(self, type, name):
        """Get a coefficient entry

        Parameters
        ----------
        type : string
            Particle type to get coefficient

        name : string
            Name of coefficient to get

        Examples
        --------
        my_coeff.get('A', 'sigma')

        Raises
        ------
        An exception if the requested parameter does not exist
        """
        return self._params[type][name]

    def verify(self):
        """Verifies all required keys are set in the dictionary

        A dictionary is valid if an entry exists for every coefficient name
        in ``require`` for every type.

        The current state of the dictionary validity may be cached to avoid
        overhead of duplicate calls during a run. The cached value is destroyed
        if an entry is modified using set().

        Returns
        -------
        __valid : bool
            True if the coefficient dictionary has all required entries

        """
        if self.require is None:
            return True
        elif self.__valid is not None:
            return self.__valid

        self.__valid = True
        try:
            for t in self.system.types:
                for r in self.require:
                    if not r in self._params[t]:
                        self.__valid = False
                        return self.__valid
        except KeyError:
            self.__valid = False

        return self.__valid

class pair_coeff(coeff):
    def __init__(self, system, require=None):
        coeff.__init__(self, system, require)

    def set(self, i, j, **coeffs):
        if isinstance(i, basestring):
            i = [i]
        if isinstance(j, basestring):
            j = [j]

        # unpack coefficients into the types
        for type_i in i:
            if not type_i in self._params:
                self._params[type_i] = {}
            for type_j in j:
                if not type_j in self._params[type_i]:
                    self._params[type_i][type_j] = {}
                for key, val in coeffs.iteritems():
                    self._params[type_i][type_j][key] = val
                    if type_i != type_j:
                        self._params[type_j][type_i][key] = val

        self.__valid = None

    def get(self, i, j, name):
        try:
            return self._params[i][j][name]
        except KeyError:
            return self._params[j][i][name]

    def verify(self):
        if self.require is None:
            return True
        elif self.__valid is not None:
            return self.__valid

        self.__valid = True
        try:
            for i in self.system.types:
                for j in self.system.types:
                    for r in self.require:
                        if not r in self._params[i][j]:
                            self.__valid = False
                            return self.__valid
        except KeyError:
            self.__valid = False

        return self.__valid
