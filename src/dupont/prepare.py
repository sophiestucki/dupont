"""Tools to bring any spectrum to the form the mask extraction needs: one wavelength medium, the resolution and the
sampling of the instrument, at rest, flattened (spectra.flatten), with or without noise.

Synthetic spectra at several mu angles (MURaM, PoET, MPS-ATLAS, ...) can be integrated over the disc with SunSim in
its high-resolution spectroscopic mode (sunsim.flux_grid(..., mode='spectroscopy', stellar_rotation=True)):
disc_integrate. SunSim is imported only by that function.
"""
import numpy as np

from .spectra import C_LIGHT, degrade_resolution, doppler_shift, vacuum_to_air  # noqa: F401 (re-exported)


# ---------------------------------------------------------------------------------------------------------------------
# range and sampling
# ---------------------------------------------------------------------------------------------------------------------
def crop(wave, *arrays, wave_range=None):
    """Spectrum (wave, *arrays) inside wave_range = (lo, hi) [A]; None entries of arrays stay None."""
    wave = np.asarray(wave, dtype=np.float64)
    if wave_range is None:
        return (wave,) + tuple(arrays)
    k = (wave >= wave_range[0]) & (wave <= wave_range[1])
    return (wave[k],) + tuple(None if a is None else np.asarray(a)[k] for a in arrays)


def velocity_grid(wave_min, wave_max, step):
    """Wavelengths [A] from wave_min to wave_max with a constant velocity step [m/s] (uniform in ln wavelength), e.g.
    the pixel size of an instrument in velocity. For an instrument with a constant step in wavelength use np.arange."""
    d = np.log1p(step / C_LIGHT)
    return np.exp(np.arange(np.log(wave_min), np.log(wave_max), d))


def resample(wave, flux, new_wave, error=None):
    """Flux-conserving resampling of (wave, flux) on the pixels new_wave (each pixel spans the mid-points to its
    neighbours): the mean flux over every new pixel, from the integral of the linear interpolation of the spectrum.
    Use it after degrade_resolution to go from a synthetic grid to the pixels of an instrument; going to a finer grid
    it is a linear interpolation. With error, the error of every new pixel is that of the mean of the old pixels it
    covers (sigma / sqrt(n_old), n_old >= 1). New pixels outside wave are NaN.
    Returns flux (and error if given)."""
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    nw = np.asarray(new_wave, dtype=np.float64)
    edges = np.concatenate([[nw[0] - 0.5 * (nw[1] - nw[0])], 0.5 * (nw[1:] + nw[:-1]), [nw[-1] + 0.5 * (nw[-1] - nw[-2])]])
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(wave))])

    def integral(x):
        # integral of the linear interpolation from wave[0] to x
        i = np.clip(np.searchsorted(wave, x) - 1, 0, len(wave) - 2)
        t = x - wave[i]
        slope = (flux[i + 1] - flux[i]) / (wave[i + 1] - wave[i])
        return cum[i] + flux[i] * t + 0.5 * slope * t ** 2

    xe = np.clip(edges, wave[0], wave[-1])
    out = np.diff(integral(xe)) / np.diff(edges)
    out[(edges[:-1] < wave[0]) | (edges[1:] > wave[-1])] = np.nan
    if error is None:
        return out
    n_old = np.maximum(np.diff(np.searchsorted(wave, edges)), 1)
    err = np.interp(nw, wave, np.asarray(error, dtype=np.float64)) / np.sqrt(n_old)
    err[~np.isfinite(out)] = np.nan
    return out, err


# ---------------------------------------------------------------------------------------------------------------------
# broadening
# ---------------------------------------------------------------------------------------------------------------------
def rotational_broadening(wave, flux, vsini, epsilon=0.6):
    """Spectrum of a rigidly rotating star from the spectrum of a non-rotating one: convolution with the rotation
    profile of Gray (2005, eq. 18.14) of half width vsini [m/s] and linear limb darkening epsilon, in velocity (on a
    uniform ln-wavelength grid at the finest step of wave, then interpolated back on wave). For a disc-integrated
    spectrum from specific intensities at several mu, prefer disc_integrate (it uses the real centre-to-limb
    variation, line by line, and differential rotation). Returns the broadened flux on wave."""
    wave = np.asarray(wave, dtype=np.float64)
    if vsini <= 0:
        return np.asarray(flux, dtype=np.float64).copy()
    lnw = np.log(wave)
    step = np.min(np.diff(lnw))
    grid = np.arange(lnw[0], lnw[-1] + 0.5 * step, step)
    fg = np.interp(grid, lnw, flux)
    n = int(np.ceil(np.log1p(vsini / C_LIGHT) / step))
    x = np.arange(-n, n + 1) * step * C_LIGHT / vsini           # v / vsini
    x = np.clip(x, -1.0, 1.0)
    k = (2 * (1 - epsilon) * np.sqrt(1 - x ** 2) + 0.5 * np.pi * epsilon * (1 - x ** 2))
    k /= k.sum()
    pad = np.concatenate([np.full(n, fg[0]), fg, np.full(n, fg[-1])])
    return np.interp(lnw, grid, np.convolve(pad, k, mode='valid'))


# ---------------------------------------------------------------------------------------------------------------------
# noise and rest frame
# ---------------------------------------------------------------------------------------------------------------------
def add_noise(flux, snr, seed=None):
    """Photon noise for a continuum S/N per pixel snr (flux normalised to a continuum of 1): sigma = sqrt(flux) / snr.
    Returns (noisy flux, sigma)."""
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.sqrt(np.clip(flux, 0, None)) / snr
    rng = np.random.default_rng(seed)
    return flux + sigma * rng.standard_normal(len(flux)), sigma


def measure_rv(wave, flux, mask_wave, mask_weight, rv=None):
    """RV [m/s] of the CCF of a flattened spectrum with a mask (gaussian fit, ccf.ccf_params)."""
    from .ccf import RV_GRID, ccf, ccf_params
    rv = RV_GRID if rv is None else rv
    return ccf_params(rv, ccf(wave, flux, mask_wave, mask_weight, rv))['rv']


def to_rest(wave, flux, mask_wave, mask_weight, rv=None):
    """Wavelengths shifted so that the CCF of the spectrum with the mask is at 0 (the frame of that mask).
    Returns (shifted wave, RV removed [m/s])."""
    v = measure_rv(wave, flux, mask_wave, mask_weight, rv)
    return doppler_shift(wave, -v), v


# ---------------------------------------------------------------------------------------------------------------------
# synthetic spectra at several mu angles -> disc-integrated spectrum (SunSim)
# ---------------------------------------------------------------------------------------------------------------------
def read_mu_spectra(path, wave_range=None):
    """Specific-intensity spectra {mu: {'wav': [A], 'intensity'}} of a .npy dictionary (MURaM / PoET high_res_grid
    files of SunSim), without the zero pixels at the edges, optionally cropped to wave_range [A]."""
    d = np.load(path, allow_pickle=True).item()
    spec = {float(mu): {'wav': np.asarray(v['wav'], dtype=np.float64), 'intensity': np.asarray(v['intensity'], dtype=np.float64)}
            for mu, v in d.items()}
    nz = np.where(np.all([s['intensity'] > 0 for s in spec.values()], axis=0))[0]
    out = {}
    for mu, s in spec.items():
        w, f = s['wav'][nz[0]:nz[-1] + 1], s['intensity'][nz[0]:nz[-1] + 1]
        w, f = crop(w, f, wave_range=wave_range)
        out[mu] = {'wav': w, 'intensity': f}
    return out


SUN = dict(rotation_period=24.47, inclination_deg=7.25)       # [days] equator, [deg] (0 = equator-on), as SunSim runs


def disc_integrate(spectra_mu, n_rings=60, rotation_period=SUN['rotation_period'], inclination_deg=SUN['inclination_deg'],
                   differential_rotation='sun', stellar_radius=1.0, wave_range=None):
    """Disc-integrated spectrum of a star from its specific intensities at >= 2 mu angles, with SunSim in its
    high-resolution spectroscopic mode: sunsim.flux_grid(..., mode='spectroscopy', stellar_rotation=True). The disc is
    cut in n_rings rings and cells; every cell takes I(mu) (linear interpolation in mu, I proportional to mu below the
    smallest mu), is Doppler shifted by its rotation velocity (solar differential rotation by default: 'sun', or
    coefficients / a law as in StarSim, 0 for rigid) and the cells are summed.

    spectra_mu: {mu: {'wav': [A], 'intensity'}} on one wavelength grid (read_mu_spectra). The wavelengths stay in
    their medium (vacuum for MURaM / PoET: vacuum_to_air afterwards). The output is shorter than the input at both
    ends by the largest rotation shift. The memory grows with the number of pixels x rings: crop to wave_range for
    tests. Returns (wave, flux) (flux not normalised: flatten it)."""
    from sunsim import flux_grid, SOLAR_DIFFERENTIAL_ROTATION
    if wave_range is not None:
        spectra_mu = {mu: dict(zip(('wav', 'intensity'), crop(s['wav'], s['intensity'], wave_range=wave_range)))
                      for mu, s in spectra_mu.items()}
    if isinstance(differential_rotation, str):
        if differential_rotation != 'sun':
            raise ValueError("differential_rotation: 'sun', 0, coefficients or a law (as StarSim)")
        differential_rotation = SOLAR_DIFFERENTIAL_ROTATION
    w = spectra_mu[max(spectra_mu)]['wav']
    g = flux_grid(n_rings, spectra_mu, 'quiet', None, w[0], w[-1], rotation_period=rotation_period,
                  inclination=np.deg2rad(inclination_deg), stellar_radius=stellar_radius, stellar_rotation=True,
                  mode='spectroscopy', differential_rotation=differential_rotation)
    g.built_grid()
    return np.asarray(g.wv, dtype=np.float64).copy(), np.asarray(g.disc_spectrum, dtype=np.float64).copy()
