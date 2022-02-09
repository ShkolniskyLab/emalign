import numpy as np
import finufft


class FINufftPlan:
    def __init__(self, sz, fourier_pts, epsilon=1e-15):
        """
        A plan for non-uniform FFT (3D)
        :param sz: A tuple indicating the geometry of the signal
        :param fourier_pts: The points in Fourier space where the Fourier transform is to be calculated,
            arranged as a 3-by-K array. These need to be in the range [-pi, pi] in each dimension.
        :param epsilon: The desired precision of the NUFFT
        """
        self.sz = sz
        self.dim = len(sz)
        # TODO: Things get messed up unless we ensure a 'C' ordering here - investigate why
        self.fourier_pts = np.asarray(np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi, order='C')
        self.num_pts = fourier_pts.shape[1]
        self.epsilon = epsilon

        # Get a handle on the appropriate 1d/2d/3d forward transform function in finufftpy
        self.transform_function = getattr(finufft, {1: 'nufft1d2', 2: 'nufft2d2', 3: 'nufft3d2'}[self.dim])
        # Get a handle on the appropriate 1d/2d/3d adjoint function in finufftpy
        self.adjoint_function = getattr(finufft, {1: 'nufft1d1', 2: 'nufft2d1', 3: 'nufft3d1'}[self.dim])

    def transform(self, signal):
        epsilon = max(self.epsilon, np.finfo(signal.dtype).eps)

        # Forward transform functions in finufftpy have signatures of the form:
        # (x, y, z, c, isign, eps, f, ...)
        # (x, y     c, isign, eps, f, ...)
        # (x,       c, isign, eps, f, ...)
        # Where f is a Fortran-order ndarray of the appropriate dimensions
        # We form these function signatures here by tuple-unpacking

        result = np.zeros(self.num_pts).astype('complex128')

        result = self.transform_function(
            self.fourier_pts[0].astype('float64'),
            self.fourier_pts[1].astype('float64'),
            self.fourier_pts[2].astype('float64'),
            signal.astype('complex128'))
            #*self.fourier_pts[0],
            #*self.fourier_pts[1],
            #*self.fourier_pts[2],
            #signal
            #*self.fourier_pts,
            #result,
            #-1,
            #epsilon,
            #signal
        #)

        #if result_code != 0:
        #    raise RuntimeError(f'FINufft transform failed. Result code {result_code}')

        return result

    def adjoint(self, signal):

        epsilon = max(self.epsilon, np.finfo(signal.dtype).eps)

        # Adjoint functions in finufftpy have signatures of the form:
        # (x, y, z, c, isign, eps, ms, mt, mu, f, ...)
        # (x, y     c, isign, eps, ms, mt      f, ...)
        # (x,       c, isign, eps, ms,         f, ...)
        # Where f is a Fortran-order ndarray of the appropriate dimensions
        # We form these function signatures here by tuple-unpacking

        # Note: Important to have order='F' here!
        result = np.zeros(self.sz, order='F').astype('complex128')

        result = self.adjoint_function(       
            self.fourier_pts[0].astype('float64'),
            self.fourier_pts[1].astype('float64'),
            self.fourier_pts[2].astype('float64'),
            signal.astype('complex128'))
            #*self.fourier_pts,
            #signal,
            #1,
            #epsilon,
            #*self.sz,
            #result
        #)
        #if result_code != 0:
        #    raise RuntimeError(f'FINufft adjoint failed. Result code {result_code}')

        return result


def cryo_project(vol, rot, n=None, eps=np.finfo(np.float32).eps, batch_size=100):
    nv = vol.shape[0]
    if n is None:
        n = nv

    r = np.arange(-(n - 1) / 2, (n - 1) / 2 + 1)
    I, J = np.meshgrid(r, r)  # i, j are reverse, but I flatten them using python indexing so they are switched back
    I = I.flatten()
    J = J.flatten()

    if n > nv + 1:
        if (n - nv) % 2 == 1:
            raise ValueError('Upsampling from odd to even sizes or vice versa is not supported')
        dN = (n - nv) // 2
        fv = cfftn(vol)
        padded_vol = np.zeros((n, n, n), dtype='complex')
        padded_vol[dN:dN+nv, dN:dN+nv, dN:dN+nv] = fv
        vol = icfftn(padded_vol).astype('float32')
        nv = n

    num_rots = rot.shape[2]
    batch_size = min(batch_size, num_rots)
    num_batches = int(np.ceil(num_rots / batch_size))
    out_projections = np.zeros((n, n, num_rots))
    for batch in range(num_batches):
        curr_batch_size = int(min(batch_size, num_rots - batch * batch_size))
        p = np.zeros((len(I) * curr_batch_size, 3))
        start = batch * batch_size

        for k in range(curr_batch_size):
            n_x = rot[0, :, start + k]
            n_y = rot[1, :, start + k]
            p[k * len(I): (k + 1) * len(I)] = np.outer(I, n_x) + np.outer(J, n_y)
        p *= -2 * np.pi / nv
        plan = FINufftPlan((nv, nv, nv), -p.T.copy(), eps)
        projection_fourier = plan.transform(vol)
        projection_fourier = np.reshape(projection_fourier, (len(I), curr_batch_size), 'F')
        p = np.reshape(p, (len(I), curr_batch_size, 3), 'F')

        if n % 2 == 0:
            projection_fourier *= np.exp(np.sum(p, 2) * (1j / 2))
            Irep = np.array([I] * curr_batch_size).T
            Jrep = np.array([J] * curr_batch_size).T
            projection_fourier *= np.exp((Irep + Jrep - 1) * (np.pi * 1j / n))

        for k in range(curr_batch_size):
            projection = np.reshape(projection_fourier[:, k], (n, n), 'F')
            projection = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(projection)))

            if n % 2 == 0:
                projection *= np.reshape(np.exp((I + J) * (np.pi * 1j / n)), (n, n), 'F')

            out_projections[:, :, start + k] = np.real(projection)

    return out_projections


def cfftn(x):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x)))


def icfftn(x):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x)))
