"""Lines common to two spectra (e.g. an observation and a simulation), their shared window and their weights."""
import numpy as np
import pandas as pd

from .spectra import C_LIGHT


def _closest_first(wa, wb, max_dv, shift=0.0):
    """One-to-one pairs (i, j) of the sorted wavelengths wa, wb with |v(b - a) - shift| <= max_dv [m/s], the closest
    pairs taken first. Every a is compared with its two neighbours in b (enough for sorted line lists)."""
    j = np.clip(np.searchsorted(wb, wa * (1.0 + shift / C_LIGHT)), 1, len(wb) - 1)
    cand = []
    for jj in (j - 1, j):
        dv = C_LIGHT * (wb[jj] - wa) / wa - shift
        ok = np.abs(dv) <= max_dv
        cand += list(zip(np.abs(dv[ok]), np.where(ok)[0], jj[ok]))
    cand.sort()
    used_a, used_b, pairs = set(), set(), []
    for _, i, k in cand:
        if i in used_a or k in used_b:
            continue
        used_a.add(i)
        used_b.add(k)
        pairs.append((i, k))
    return sorted(pairs)


def pair_lines(la, lb, max_dv=2000.0, shift=None):
    """Lines common to two line tables (detect_lines): one-to-one pairs of minima, closest first, within max_dv [m/s]
    of each other once a global velocity shift between the two spectra is removed. shift None: the median velocity
    difference of the pairs found within 3 max_dv (the two spectra need not be on the same RV zero point); a number:
    that shift [m/s]. Returns (common_a, common_b, shift): row k of both tables is the same line, sorted by wavelength,
    with 'dv' = velocity of b - a minus the shift [m/s]."""
    la = la.sort_values('wave_min').reset_index(drop=True)
    lb = lb.sort_values('wave_min').reset_index(drop=True)
    wa, wb = la['wave_min'].values, lb['wave_min'].values
    if shift is None:
        first = _closest_first(wa, wb, 3 * max_dv)
        shift = float(np.median([C_LIGHT * (wb[k] - wa[i]) / wa[i] for i, k in first])) if first else 0.0
    pairs = _closest_first(wa, wb, max_dv, shift)
    ia, ib = (np.array([p[q] for p in pairs], dtype=int) for q in (0, 1))
    ca, cb = la.iloc[ia].reset_index(drop=True), lb.iloc[ib].reset_index(drop=True)
    dv = C_LIGHT * (cb['wave_min'].values - ca['wave_min'].values) / ca['wave_min'].values - shift
    ca['dv'] = cb['dv'] = dv
    return ca, cb, shift


def shared_window(ca, cb, wave_a, wave_b, how='min'):
    """The same window of pixels for both lines of every pair: 'min' from the innermost to the innermost maximum
    (intersection of the two windows), 'max' the union; 'own' keeps both windows. The first windows are kept in
    wave_left_max_own, wave_right_max_own. Returns the two tables."""
    if how == 'own':
        return ca, cb
    pick_l, pick_r = (np.maximum, np.minimum) if how == 'min' else (np.minimum, np.maximum)
    wl = pick_l(ca['wave_left_max'].values, cb['wave_left_max'].values)
    wr = pick_r(ca['wave_right_max'].values, cb['wave_right_max'].values)
    ok = wr > wl
    ca, cb = ca[ok].reset_index(drop=True), cb[ok].reset_index(drop=True)
    wl, wr = wl[ok], wr[ok]
    out = []
    for t, wave in ((ca, wave_a), (cb, wave_b)):
        t = t.copy()
        t['wave_left_max_own'], t['wave_right_max_own'] = t['wave_left_max'], t['wave_right_max']
        t['i_left'] = np.clip(np.searchsorted(wave, wl), 0, len(wave) - 1)
        t['i_right'] = np.clip(np.searchsorted(wave, wr), 0, len(wave) - 1)
        t['wave_left_max'], t['wave_right_max'] = wave[t['i_left']], wave[t['i_right']]
        out.append(t)
    return tuple(out)


def rv_weights(wave, flux, error=None, lines=None, detector_noise=0.0):
    """RV information of every line, Bouchy et al. (2001) eq. 8:
        W = sum over the pixels of the line window (i_left..i_right) of lambda^2 (dA0/dlambda)^2 / (A0 + sigma_D^2).
    By default (error None) A0 is the flattened flux itself (continuum = 1) and sigma_D the detector noise in the same
    units (detector_noise, default 0): the photon noise of each pixel is set by its own flux, so the line core counts
    less than the wings, and the weight depends only on the spectrum, not on the S/N of the observation. With error
    given, the denominator is error^2 instead (the measured variance, A0 + sigma_D^2 of the observation in the flux
    units): this W gives the photon-noise RV error of that observation, sigma_RV = c / sqrt(W).
    Pixels with an undefined derivative or variance count 0. Adds 'weight' and 'sigma_rv' [m/s] (sigma_rv is in m/s
    only with error given; otherwise it is c / sqrt(W) for a continuum of one photo-electron)."""
    if lines is None:
        raise TypeError('rv_weights needs the line table (lines)')
    flux = np.asarray(flux, dtype=np.float64)
    if error is None:
        var = flux + detector_noise ** 2
    else:
        var = np.broadcast_to(np.asarray(error, dtype=np.float64), np.shape(flux)) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        w = wave ** 2 * np.gradient(flux, wave) ** 2 / var #todocheck for poet with sigma_D=0
    w = np.where(np.isfinite(w) & (var > 0), w, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(w)])
    out = lines.copy()
    out['weight'] = cum[out['i_right'].astype(int).values + 1] - cum[out['i_left'].astype(int).values]
    with np.errstate(divide='ignore'):
        out['sigma_rv'] = C_LIGHT / np.sqrt(out['weight'].values)
    return out


def match_reference(lines_wave, ref_wave, max_dwave=0.11):
    """One-to-one match of a reference mask (e.g. the ESPRESSO mask, wavelengths ref_wave) to lines (lines_wave):
    pairs within max_dwave [A], the closest pairs first, so that no line takes two reference lines and no reference line
    two lines. Returns a DataFrame (ref_index, line_index, dwave = line - reference), sorted by ref_index."""
    lw = np.asarray(lines_wave, dtype=np.float64)
    rw = np.asarray(ref_wave, dtype=np.float64)
    order = np.argsort(lw)
    ls = lw[order]
    j = np.clip(np.searchsorted(ls, rw), 1, len(ls) - 1)
    cand = []
    for jj in (j - 1, j):
        d = ls[jj] - rw
        ok = np.abs(d) <= max_dwave
        cand += list(zip(np.abs(d[ok]), np.where(ok)[0], order[jj[ok]], d[ok]))
    cand.sort()
    used_r, used_l, rows = set(), set(), []
    for _, r, l, d in cand:
        if r in used_r or l in used_l:
            continue
        used_r.add(r)
        used_l.add(l)
        rows.append((r, l, d))
    return pd.DataFrame(sorted(rows), columns=['ref_index', 'line_index', 'dwave'])
