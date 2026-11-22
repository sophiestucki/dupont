"""Cross-correlation of a spectrum with a line mask, and the RV, FWHM, contrast and bisector span of the CCF.

The CCF follows SunSim (sunsim.nbspectra.cross_correlation_mask, any wavelength grid): every mask line, Doppler shifted,
takes the flux over a box of one pixel centred on it (split between the two pixels it overlaps, linear interpolation),
times its weight; the CCF is minus the weighted sum, so it peaks at the velocity of the lines.
"""
import numpy as np
from scipy import optimize

from .spectra import C_LIGHT

RV_GRID = 1e3 * np.arange(-18.0, 18.25, 0.25)  # [m/s]


def ccf(wave, flux, mask_wave, mask_weight, rv=RV_GRID):
    """CCF (not normalised) of the spectrum (wave, flux) with the mask (mask_wave [A], mask_weight) at the velocities
    rv [m/s]. Lines that fall outside the spectrum do not count."""
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    mw = np.asarray(mask_wave, dtype=np.float64)
    mf = np.asarray(mask_weight, dtype=np.float64)
    out = np.zeros(len(rv))
    for i, v in enumerate(rv):
        wl = mw * (1.0 + v / C_LIGHT)
        ok = (wl >= wave[0]) & (wl < wave[-1])
        wl, f_m = wl[ok], mf[ok]
        il = np.searchsorted(wave, wl, side='right') - 1
        ir = il + 1
        dw = wave[ir] - wave[il]
        mid = 0.5 * (wave[ir] + wave[il])
        left, right = wl - 0.5 * dw, wl + 0.5 * dw
        frac1, frac2 = (mid - left) / dw, (right - mid) / dw
        slope = (flux[ir] - flux[il]) / dw
        f1 = flux[il] + (0.5 * (left + mid) - wave[il]) * slope
        f2 = flux[il] + (0.5 * (right + mid) - wave[il]) * slope
        out[i] = -np.sum(f1 * f_m * frac1 + f2 * f_m * frac2)
    return out


def _wings(rv, c):
    """Edges (cutleft, cutright) of a CCF peak normalised to its maximum: where it starts to rise again on each side,
    below 80 % of the maximum (as sunsim.nbspectra.speed_bisector_nb)."""
    imax = int(np.argmax(c))
    cutleft, cutright = 0, len(c) - 1
    for i in range(imax + 1, len(c)):
        if c[i] > c[i - 1] and c[i] < 0.8 * c[imax]:
            cutright = i
            break
    for i in range(imax - 1, 0, -1):
        if c[i - 1] > c[i] and c[i - 1] < 0.8 * c[imax]:
            cutleft = i
            break
    return cutleft, cutright


def bisector(rv, c, heights, cutleft, cutright):
    """Bisector of the CCF peak c (normalised to its maximum) at the given heights, between the wings: mid-point of the
    two velocities where c crosses every height, the crossings closest to the maximum (linear interpolation, as
    sunsim.nbspectra.bisector_at_nb)."""
    x, y = rv[cutleft:cutright], c[cutleft:cutright]
    xmax = rv[int(np.argmax(c))]
    out = np.full(len(heights), np.nan)
    for k, h in enumerate(heights):
        jl = np.where((y[:-1] < h) & (y[1:] > h) & (x[:-1] < xmax))[0]
        jr = np.where((y[:-1] > h) & (y[1:] < h) & (x[1:] > xmax))[0]
        if len(jl) and len(jr):
            a, b = jl[-1], jr[-1]
            v1 = x[a] + (x[a + 1] - x[a]) * (h - y[a]) / (y[a + 1] - y[a])
            v2 = x[b] + (x[b + 1] - x[b]) * (h - y[b]) / (y[b + 1] - y[b])
            out[k] = 0.5 * (v1 + v2)
    return out


def ccf_params(rv, c, vsini=2052.9, window=None, bis_heights='bisector'):
    """RV, contrast and FWHM (gaussian fit) and BIS (mean bisector at 10-40 % of the height minus 60-90 %) of a CCF
    from ccf(). window: fixed (cutleft, cutright) indices of rv for the fit and the bisector, instead of the wings of
    this CCF. bis_heights: 'bisector', the BIS from the 50 heights of the bisector (as making_mask.ipynb); 'fixed', from
    31 fixed heights in each band (as sunsim.nbspectra.bisector_span_nb). Returns a dict (rv, contrast, fwhm, bis
    [m/s], x_bis, y_bis, cutleft, cutright)."""
    c = np.asarray(c, dtype=np.float64)
    c = c - c.min() + 1e-6
    cn = c / c.max()
    cutleft, cutright = _wings(rv, cn) if window is None else window
    x, y = rv[cutleft:cutright], cn[cutleft:cutright]
    xmax = rv[int(np.argmax(cn))]
    minccf = max(np.min(y[x > xmax]), np.min(y[x < xmax]))
    y_bis = np.linspace(minccf + 0.1 * (1 - minccf), 0.99, 50)
    x_bis = bisector(rv, cn, y_bis, cutleft, cutright)
    if bis_heights == 'fixed':
        lo = max(0.1, minccf + 0.05 * (1 - minccf))
        bis = (np.mean(bisector(rv, cn, np.linspace(lo, 0.4, 31), cutleft, cutright))
               - np.mean(bisector(rv, cn, np.linspace(0.6, 0.9, 31), cutleft, cutright)))
    else:
        bis = (np.nanmean(x_bis[(y_bis >= 0.1) & (y_bis <= 0.4)]) - np.nanmean(x_bis[(y_bis >= 0.6) & (y_bis <= 0.9)]))
    x, y = rv[cutleft:cutright], c[cutleft:cutright]
    g = lambda x, a, mu, s, off: off + a * np.exp(-(x - mu) ** 2 / (2 * s ** 2))
    popt, _ = optimize.curve_fit(g, x, y, p0=[np.max(y), x[np.argmax(y)] + 100, 1.5 * vsini + 1000, 1e-6], maxfev=5000)
    return {'rv': popt[1], 'contrast': popt[0], 'fwhm': 2 * np.sqrt(2 * np.log(2)) * abs(popt[2]), 'bis': bis,
            'x_bis': x_bis, 'y_bis': y_bis, 'cutleft': cutleft, 'cutright': cutright}


def photon_rv_error(weights_sigma_rv):
    """Photon-noise RV uncertainty [m/s] of a mask from the RV uncertainties of its lines (rv_weights 'sigma_rv' with
    the measured error, e.g. build_mask 'obs_sigma_rv_stack'): 1 / sqrt(sum 1 / sigma_i^2) (Bouchy et al. 2001)."""
    s = np.asarray(weights_sigma_rv, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    return 1.0 / np.sqrt(np.sum(1.0 / s ** 2))
