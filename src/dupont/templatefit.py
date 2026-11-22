"""CCF indicators from a second-order template fit: every CCF is fitted with a template and its first two derivatives
(shift, depth change, broadening), which is more precise than a gaussian fit and bisector of every CCF alone."""
import numpy as np
from scipy.interpolate import CubicSpline

from .ccf import RV_GRID, _wings, ccf_params


def template_reference(P, rv=RV_GRID):
    """Template of the fit from the CCF P (a peak, from ccf()): T = 1 - P in absorption, its derivatives, the window
    between the wings and its parameters (ccf_params); s: sigma of its gaussian fit."""
    rv = np.asarray(rv, dtype=np.float64)
    c0 = P - P.min() + 1e-6
    cl, cr = _wings(rv, c0 / c0.max())
    w = np.zeros(len(rv), dtype=bool)
    w[int(cl):int(cr)] = True
    q = ccf_params(rv, P)
    ref = dict(rv=rv, P=P, window=w, rv0=q['rv'], con0=q['contrast'], fwhm0=q['fwhm'], bis0=q['bis'], xbis0=q['x_bis'],
               ybis0=q['y_bis'], s=q['fwhm'] / (2 * np.sqrt(2 * np.log(2))), T=1 - P)
    ref['T1'] = CubicSpline(rv, ref['T'])(rv, 1)
    ref['T2'] = CubicSpline(rv, ref['T1'])(rv, 1)
    return ref


def template_fit(ref, ccfs, heights=None, offset=True):
    """Fit of every CCF (rows of ccfs) with the template, on its window, in absorption (1 - CCF):
    C = T + a (T - 1) - drv T' + s_T ds T'' (+ a constant with offset). The T'' term is a broadening at constant area
    (sigma -> sigma + ds), which also lowers the depth by ds / s_T. The bisector: shift of each flank at a fixed relative
    height; BIS: mean over 10-40 % minus 60-90 % of the height. Returns a dict: a, drv, ds, rv, contrast, fwhm, bis
    (comparable with ccf_params) and the bisector at `heights` (default: those of the template)."""
    v, T, T1, T2, window, s_T = ref['rv'], ref['T'], ref['T1'], ref['T2'], ref['window'], ref['s']
    heights = ref['ybis0'] if heights is None else heights
    r = (1 - np.atleast_2d(ccfs)) - T
    X = np.column_stack([T[window] - 1, -T1[window], s_T * T2[window]] + ([np.ones(window.sum())] if offset else []))
    a, drv, ds = np.linalg.lstsq(X, r[:, window].T, rcond=None)[0][:3]
    k = 1 + a - ds / s_T
    i_top, i_core = np.argmax(T), np.argmin(T)
    y = (T[i_top] - T) / (T[i_top] - T[i_core])
    D = (r - r[:, [i_top]] - y * (r[:, [i_core]] - r[:, [i_top]])) / k[:, None]
    left, right = window & (v < v[i_core]), window & (v > v[i_core])

    def band(flank, lo, hi):
        sel = flank & (y >= lo) & (y <= hi)
        return -np.sum(np.sign(T1[sel]) * D[:, sel], axis=1) / np.sum(np.abs(T1[sel]))

    def bis_band(lo, hi):
        return 0.5 * (band(left, lo, hi) + band(right, lo, hi))

    def flank_shift(flank):
        sel = flank & (y > 0.02) & (y < 0.995)
        o = np.argsort(y[sel])
        dv = -D[:, sel] / T1[sel]
        return np.array([np.interp(heights, y[sel][o], d[o]) for d in dv])

    minccf = (ref['ybis0'][0] - 0.1) / 0.9
    bis = bis_band(max(0.1, minccf + 0.05 * (1 - minccf)), 0.4) - bis_band(0.6, 0.9)
    bisector = 0.5 * (flank_shift(left) + flank_shift(right))
    return dict(a=a, drv=drv, ds=ds, rv=ref['rv0'] + drv / k, contrast=ref['con0'] * k,
                fwhm=ref['fwhm0'] * (1 + ds / k / s_T), bis=ref['bis0'] + bis,
                bisector=np.interp(heights, ref['ybis0'], ref['xbis0']) + bisector)


def template_fit_iterative(ccfs, rv=RV_GRID, n_iter=5):
    """Template fit with a template built iteratively: start from the median CCF; at every iteration fit all the CCFs,
    shift each to the template RV, and take the median of the aligned CCFs as the new template. Returns the fit of the
    last iteration and the history (list of dicts: iteration, std of rv, fwhm, bis, rms change of the template)."""
    rv = np.asarray(rv, dtype=np.float64)
    cs = np.asarray(ccfs, dtype=np.float64)
    P, P_old, hist = np.median(cs, axis=0), None, []
    for it in range(n_iter + 1):
        ref = template_reference(P, rv)
        t = template_fit(ref, cs)
        hist.append({'iteration': it, 'rv std': np.std(t['rv']), 'fwhm std': np.std(t['fwhm']), 'bis std': np.std(t['bis']),
                     'template change': np.nan if P_old is None else float(np.sqrt(np.mean((P - P_old) ** 2)) / np.ptp(P))})
        if it == n_iter:
            break
        shift = t['rv'] - ref['rv0']
        aligned = np.array([CubicSpline(rv, c)(rv + d) for c, d in zip(cs, shift)])
        P_old, P = P, np.median(aligned, axis=0)
    return t, hist
