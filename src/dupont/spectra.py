"""Reading, stacking, resolution and continuum normalisation of spectra.

Every spectrum is a pair (or triplet) of numpy arrays: wavelengths [A], flux and, when known, the 1-sigma error of
the flux. Wavelengths are in air unless a function says otherwise.
"""
import os

import numpy as np
import pandas as pd

C_LIGHT = 299792458.0  # [m/s]


# ---------------------------------------------------------------------------------------------------------------------
# wavelengths
# ---------------------------------------------------------------------------------------------------------------------
def vacuum_to_air(wave):
    """Air wavelengths [A] of the vacuum wavelengths `wave` [A] (Ciddor 1996 dispersion formula, as in SunSim)."""
    s2 = (1e4 / np.asarray(wave, dtype=np.float64)) ** 2
    n = 1.0 + 5.792105e-2 / (238.0185 - s2) + 1.67917e-3 / (57.362 - s2)
    return wave / n


def doppler_shift(wave, rv):
    """Wavelengths `wave` Doppler shifted by the velocity rv [m/s] (positive: redshift)."""
    return np.asarray(wave, dtype=np.float64) * (1.0 + rv / C_LIGHT)


def common_grid(wave_a, flux_a, wave_b, flux_b, error_a=None, error_b=None):
    """Two spectra on the same wavelengths: the range common to both, spectrum b interpolated linearly on the
    wavelengths of a (unless they are already the same), and every pixel that is not finite in one of the two removed
    from both. Returns (wave, flux_a, flux_b, error_a, error_b) (errors None when not given)."""
    lo, hi = max(wave_a.min(), wave_b.min()), min(wave_a.max(), wave_b.max())
    ka, kb = (wave_a >= lo) & (wave_a <= hi), (wave_b >= lo) & (wave_b <= hi)
    wave = wave_a[ka]
    fa = flux_a[ka]
    ea = None if error_a is None else error_a[ka]
    if len(wave_b[kb]) == len(wave) and np.array_equal(wave_b[kb], wave):
        fb = flux_b[kb]
        eb = None if error_b is None else error_b[kb]
    else:
        ok = np.isfinite(flux_b[kb])
        fb = np.interp(wave, wave_b[kb][ok], flux_b[kb][ok], left=np.nan, right=np.nan)
        eb = None if error_b is None else np.interp(wave, wave_b[kb][ok], error_b[kb][ok], left=np.nan, right=np.nan)
    ok = np.isfinite(fa) & np.isfinite(fb)
    sub = lambda x: None if x is None else x[ok]
    return wave[ok], fa[ok], fb[ok], sub(ea), sub(eb)


# ---------------------------------------------------------------------------------------------------------------------
# ESPRESSO S1D spectra
# ---------------------------------------------------------------------------------------------------------------------
def read_s1d(path, wave_col='wavelength_air'):
    """ESPRESSO S1D spectrum: wavelengths (air by default), flux and its error [e-] of the good pixels (quality = 0,
    finite, flux > 0), and header values (BJD, BERV, DRS CCF RV/FWHM/BIS/contrast, mask, instrument mode).
    Returns (wave, flux, error, info)."""
    from astropy.io import fits
    with fits.open(path) as h:
        d, hd = h[1].data, h[0].header
        w, f, e, q = (np.asarray(d[c], dtype=np.float64) for c in (wave_col, 'flux', 'error', 'quality'))
    ok = (q == 0) & np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0)
    get = lambda k, scale=1.0: scale * float(hd['HIERARCH ESO ' + k]) if 'HIERARCH ESO ' + k in hd else np.nan
    info = {'file': os.path.basename(path), 'date': hd.get('DATE-OBS'), 'mode': hd.get('HIERARCH ESO INS MODE'),
            'exptime': hd.get('EXPTIME'), 'bjd': get('QC BJD'), 'berv': get('QC BERV', 1e3),
            'drs_rv': get('QC CCF RV', 1e3), 'drs_fwhm': get('QC CCF FWHM', 1e3), 'drs_bis': get('QC CCF BIS SPAN', 1e3),
            'drs_contrast': get('QC CCF CONTRAST'), 'drs_mask': hd.get('HIERARCH ESO QC CCF MASK')}
    return w[ok], f[ok], e[ok], info


def stack_s1d(paths, wave_col='wavelength_air'):
    """Mean of several S1D spectra on the same wavelength grid (ESPRESSO S1D files of one night share it).

    Every spectrum is first scaled to the median flux of the first one (clouds and airmass change the flux level, not
    the line shapes); the flux is the mean of the scaled spectra and the error is propagated,
    sigma = sqrt(sum sigma_i^2) / N. Only the pixels good in every spectrum are kept. The spectra are not shifted:
    the solar RV changes by a few m/s over a night, far below a pixel.
    Returns (wave, flux, error, info table)."""
    from astropy.io import fits
    waves, fluxes, errors, good, infos = None, [], [], None, []
    for path in paths:
        with fits.open(path) as h:
            d = h[1].data
            w = np.asarray(d[wave_col], dtype=np.float64)
            f, e, q = (np.asarray(d[c], dtype=np.float64) for c in ('flux', 'error', 'quality'))
        if waves is None:
            waves = w
        elif not np.array_equal(w, waves):
            raise ValueError('%s is not on the wavelength grid of %s' % (path, paths[0]))
        ok = (q == 0) & np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0)
        good = ok if good is None else good & ok
        fluxes.append(f)
        errors.append(e)
        infos.append(read_s1d_header(path))
    fluxes, errors = np.array(fluxes), np.array(errors)
    ref = np.median(fluxes[0][good])
    scale = ref / np.array([np.median(f[good]) for f in fluxes])
    n = len(fluxes)
    flux = np.sum(fluxes * scale[:, None], axis=0) / n
    error = np.sqrt(np.sum((errors * scale[:, None]) ** 2, axis=0)) / n
    info = pd.DataFrame(infos)
    info['scale'] = scale
    return waves[good], flux[good], error[good], info


def read_s1d_header(path):
    """Header values of an S1D file (the info of read_s1d without reading the data)."""
    from astropy.io import fits
    hd = fits.getheader(path, 0)
    get = lambda k, scale=1.0: scale * float(hd['HIERARCH ESO ' + k]) if 'HIERARCH ESO ' + k in hd else np.nan
    return {'file': os.path.basename(path), 'date': hd.get('DATE-OBS'), 'mode': hd.get('HIERARCH ESO INS MODE'),
            'exptime': hd.get('EXPTIME'), 'bjd': get('QC BJD'), 'berv': get('QC BERV', 1e3),
            'drs_rv': get('QC CCF RV', 1e3), 'drs_fwhm': get('QC CCF FWHM', 1e3), 'drs_bis': get('QC CCF BIS SPAN', 1e3),
            'drs_contrast': get('QC CCF CONTRAST'), 'drs_mask': hd.get('HIERARCH ESO QC CCF MASK')}


# nominal resolving power of the ESPRESSO modes (ESO user manual)
ESPRESSO_RESOLUTION = {'SINGLEUHR': 190000.0, 'SINGLEHR': 140000.0, 'MULTIMR': 70000.0, 'SINGLEMR': 70000.0}


# ---------------------------------------------------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------------------------------------------------
def degrade_resolution(wave, flux, resolution):
    """Spectrum convolved with a gaussian instrumental profile of constant FWHM = c / resolution in velocity (its width
    in A grows with the wavelength). The spectrum is resampled on a uniform grid in ln(wavelength) at its finest step,
    convolved, and interpolated back on `wave`. Returns the convolved flux on `wave`."""
    from scipy.ndimage import gaussian_filter1d
    wave = np.asarray(wave, dtype=np.float64)
    lnw = np.log(wave)
    step = np.min(np.diff(lnw))
    grid = np.arange(lnw[0], lnw[-1] + 0.5 * step, step)
    fg = np.interp(grid, lnw, flux)
    sigma = 1.0 / resolution / (2.0 * np.sqrt(2.0 * np.log(2.0))) / step
    return np.interp(lnw, grid, gaussian_filter1d(fg, sigma, mode='nearest'))


# ---------------------------------------------------------------------------------------------------------------------
# continuum
# ---------------------------------------------------------------------------------------------------------------------
# blue and red parts fitted separately (cut at 4200 A), SNT alpha-shape radius 50-200 (keeps the wings of the broad
# lines; flattening_test.ipynb)
FLATTEN = dict(cut_wave=4200.0, overlap=5.0, fwhm_blue=0.1, fwhm_red=0.5, snt_config={'radius_min': 50, 'radius_max': 200})


def continuum_snt(wave, flux, cut_wave=FLATTEN['cut_wave'], overlap=FLATTEN['overlap'], fwhm_blue=FLATTEN['fwhm_blue'],
                  fwhm_red=FLATTEN['fwhm_red'], snt_config=FLATTEN['snt_config']):
    """Continuum of a spectrum with SNT (https://github.com/iastro-pt/Spectra-Normalization-Tool), the parts below and
    above cut_wave [A] fitted separately (each `overlap` [A] beyond the cut: SNT leaves NaN at the edges) and merged.
    cut_wave None: the whole spectrum at once (fwhm_red). The continuum is NaN where SNT gives none."""
    from SNT import normalize_spectra
    import loguru
    loguru.logger.disable('SNT')

    def part(sel, fwhm):
        return normalize_spectra(wavelengths=wave[sel], spectra=flux[sel], FWHM_override=fwhm, header={}, output_path='',
                                 store_to_disk=False, fname='', user_config=snt_config)[0]
    if cut_wave is None or not np.any(wave < cut_wave) or not np.any(wave >= cut_wave):
        return part(np.ones(len(wave), bool), fwhm_red)
    blue = wave < cut_wave
    cont_blue = part(wave < cut_wave + overlap, fwhm_blue)
    cont_red = part(wave >= cut_wave - overlap, fwhm_red)
    cont = np.empty_like(flux, dtype=np.float64)
    cont[blue] = cont_blue[:np.sum(blue)]
    cont[~blue] = cont_red[-np.sum(~blue):]
    return cont


def flatten(wave, flux, error=None, continuum=None, **snt):
    """Flattened spectrum: flux / continuum (continuum_snt unless given), and the error divided by the same continuum.
    The pixels without continuum are removed. Returns (wave, flux, error) (error None when not given)."""
    cont = continuum_snt(wave, flux, **snt) if continuum is None else continuum
    ok = np.isfinite(cont) & (cont > 0)
    return wave[ok], flux[ok] / cont[ok], None if error is None else error[ok] / cont[ok]


# ---------------------------------------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------------------------------------
def save_spectrum(path, wave, flux, error=None):
    """csv with columns wave, flux (and error)."""
    d = {'wave': wave, 'flux': flux}
    if error is not None:
        d['error'] = error
    pd.DataFrame(d).to_csv(path, index=False)


def load_spectrum(path):
    """(wave, flux, error) of a csv of save_spectrum (error None if the file has none)."""
    d = pd.read_csv(path)
    return d['wave'].values, d['flux'].values, d['error'].values if 'error' in d else None
