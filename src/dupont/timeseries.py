"""CCF time series of many spectra with several masks, and simple comparisons of the masks."""
import numpy as np
import pandas as pd

from .ccf import RV_GRID, ccf, ccf_params


def ccf_timeseries(spectra, masks, rv=RV_GRID, info=None):
    """CCF of every spectrum with every mask. spectra: list of (wave, flux) flattened; masks: {name: DataFrame (wave,
    weight)}; info: optional list of dicts (one per spectrum, e.g. read_s1d info) copied to the table.
    Returns a table (one row per spectrum, <mask>_rv, _fwhm, _bis, _contrast) and {mask: array (n_spectra, n_rv)}."""
    rows, ccfs = [], {m: [] for m in masks}
    for k, (w, f) in enumerate(spectra):
        row = dict(info[k]) if info is not None else {}
        for name, m in masks.items():
            c = ccf(w, f, m['wave'].values, m['weight'].values, rv)
            q = ccf_params(rv, c)
            ccfs[name].append(c)
            row.update({'%s_%s' % (name, key): q[key] for key in ('rv', 'fwhm', 'bis', 'contrast')})
        rows.append(row)
    return pd.DataFrame(rows), {m: np.array(v) for m, v in ccfs.items()}


def summary(table, masks, keys=('rv', 'fwhm', 'bis')):
    """Median and standard deviation of every indicator for every mask."""
    out = {}
    for m in masks:
        out[m] = {}
        for k in keys:
            x = table['%s_%s' % (m, k)].values
            out[m]['%s median' % k.upper()], out[m]['%s std' % k.upper()] = np.median(x), np.std(x)
    return pd.DataFrame(out).T


def differential_scatter(table, masks, key='rv'):
    """Standard deviation of the difference of an indicator between every pair of masks. The solar signal common to
    both masks (p-modes, granulation) cancels in the difference, so what is left is the noise and the part of the signal
    that the two masks see differently. Returns a square DataFrame [m/s]."""
    names = list(masks)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            if a != b:
                out.loc[a, b] = np.std(table['%s_%s' % (a, key)].values - table['%s_%s' % (b, key)].values)
    return out


def line_presence(spectra, mask_wave, reference=None, prominence=0.05, ref_prominence=None, max_dv=1000.0, shift=0.0):
    """Fraction of the spectra in which every mask line is detected. spectra: list of (wave, flux) flattened, at the
    velocity `shift` [m/s] relative to the mask. With reference (wave, flux), the spectrum the mask was built from in
    the mask frame, every mask line gets the window (between the two maxima) of the closest reference minimum within
    max_dv (detected with ref_prominence, default prominence); the line is present in a spectrum when the spectrum has
    a minimum (detect_lines with prominence, minimum pixel) inside that window. A single exposure is noisier than a
    stack, so its minimum can move by a few pixels or split in two: the window, and a prominence lower than the one
    of the mask, make the test about the line being there, not about its exact minimum. Without reference, or for a
    mask line with no reference minimum: a minimum within max_dv of the mask wavelength. Returns an array (one per
    line)."""
    from .lines import detect_lines
    c = 299792458.0
    mw = np.asarray(mask_wave, dtype=np.float64)
    lo, hi = mw * (1 - max_dv / c), mw * (1 + max_dv / c)
    if reference is not None:
        r = detect_lines(reference[0], reference[1], prominence=ref_prominence or prominence, center='pixel')
        r = r.sort_values('wave_min').reset_index(drop=True)
        rw = r['wave_min'].values
        j = np.clip(np.searchsorted(rw, mw), 1, len(rw) - 1)
        j = np.where(np.abs(rw[j - 1] - mw) < np.abs(rw[j] - mw), j - 1, j)
        near = np.abs(rw[j] - mw) < mw * max_dv / c
        lo = np.where(near, r['wave_left_max'].values[j], lo)
        hi = np.where(near, r['wave_right_max'].values[j], hi)
    lo, hi = lo * (1 + shift / c), hi * (1 + shift / c)
    found = np.zeros(len(mw))
    for w, f in spectra:
        d = np.sort(detect_lines(w, f, prominence=prominence, center='pixel')['wave_min'].values)
        found += np.searchsorted(d, hi, side='right') > np.searchsorted(d, lo, side='left')
    return found / len(spectra)
