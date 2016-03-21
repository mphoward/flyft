import numpy as np

__all__ = ['convolve']

def convolve(f,g,fft=True, periodic=True):
    """Compute the linear or circular convolution of two discrete data sets.

    For small problems, it is usually faster to perform the convolution
    directly. In this case, `convolve` defaults to the ``numpy.convolve``
    routine. For circular convolution, a copy of the smaller of the
    two arrays must be made. For larger problems, convolution is more
    efficiently performed as a multiplication in Fourier space followed by
    inversion. These transforms are performed using Fast Fourier Transform
    (FFT) methods. FFTs are fastest for intelligently chosen problem sizes
    (for example, powers of 2). If convolution performance is critical, both
    methods should be timed, and the optimal one selected.

    For linear convolutions with FFTs, `f` and `g` are zero-padded to
    ``len(f)+len(g)-1``. For circular convolutions, the period is taken to be
    the longer of `f` and `g`. If one is shorter than the other, it is
    zero-padded with ``numpy.pad``.

    Parameters
    ----------
    f : array_like
        First function to convolve

    g : array_like
        Second function to convolve

    fft : bool, optional
        If true, perform the convolution using FFTs.

    periodic : bool, optional
        If true, perform the circular convolution (periodic data).

    Returns
    -------
    out : ndarray
        The convolution of `f` and `g`

    Examples
    --------
    Take the linear convolution of two boxcar functions of even and odd lengths.
    >>> f = np.ones(5)
    >>> g = np.ones(2)
    >>> flyft.fft.convolve(f,g,periodic=False)
    [ 1.  2.  2.  2.  2.  1.]

    The convolution is returned in the proper order.
    >>> g[1] = 2
    >>> flyft.fft.convolve(f,g,periodic=False)
    [ 1.  3.  3.  3.  3.  2.]

    The convolution operator commutes.
    >>> flyft.fft.convolve(g,f,periodic=False)
    [ 1.  3.  3.  3.  3.  2.]

    """

    if not fft:
        if not periodic:
            return np.convolve(f,g)
        else:
            # replicate the smaller data set for circular convolution without FFT
            if len(f) < len(g):
                ff = np.concatenate((f,f))
                gg = g
                min_idx = len(f)
                max_idx = -len(f)+1
            else:
                ff = f
                gg = np.concatenate((g,g))
                min_idx = len(g)
                max_idx = -len(g)+1
            c = np.convolve(ff,gg)
            period = max(len(f),len(g))
            return c[period:-period+1]
    else:
        if not periodic:
            # the minimum length needed to pad to for linear convolution
            input_len = len(f) + len(g) - 1
        else:
            input_len = max(len(f),len(g))

        # fft the data
        F = np.fft.fft(f, n=input_len)
        G = np.fft.fft(g, n=input_len)

        # multiply and take the inverse
        c = np.fft.ifft(F*G)

        # extract the relevant parts
        return np.real(c)
