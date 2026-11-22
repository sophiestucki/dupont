"""Detection of the absorption lines of a flattened spectrum and their identification in a line list (VALD)."""
import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.signal import find_peaks

CORE = 0.2  # core of a line: pixels with flux < minimum + CORE x depth


# ---------------------------------------------------------------------------------------------------------------------
# minimum of a line
# ---------------------------------------------------------------------------------------------------------------------
def line_core(flux, i, i_left, i_right, core=CORE):
    """Pixels lo..hi (included) of the core of the line of minimum pixel i: around i while the flux stays below
    flux[i] + core x depth (depth: mean of the two maxima - minimum), at least i-1..i+1, inside the maxima."""
    f_max = flux[i] + core * (0.5 * (flux[i_left] + flux[i_right]) - flux[i])
    lo, hi = i - 1, i + 1
    while lo > i_left and flux[lo - 1] < f_max:
        lo -= 1
    while hi < i_right and flux[hi + 1] < f_max:
        hi += 1
    return lo, hi


def _gaussian(x, a, mu, s, cont):
    return cont - a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def refine_minimum(wave, flux, i, i_left, i_right, center='gaussian', core=CORE):
    """Minimum of the line of minimum pixel i from a fit of its core (line_core): a parabola, or a gaussian under the
    mean of the two maxima (continuum fixed). Returns (wave, flux, ok, number of pixels of the fit); ok False (the
    minimum pixel is kept) when the fit fails, has no minimum, or puts it outside the core."""
    lo, hi = line_core(flux, i, i_left, i_right, core)
    x0, cont = wave[i], 0.5 * (flux[i_left] + flux[i_right])
    x, y = wave[lo:hi + 1] - x0, flux[lo:hi + 1]
    w = np.nan
    if center == 'parabola':
        p = np.polyfit(x, y, 2)
        if p[0] > 0:
            xm = -p[1] / (2 * p[0])
            w, f = x0 + xm, np.polyval(p, xm)
    elif center == 'gaussian':
        s0 = max(0.5 * (x[-1] - x[0]), wave[i + 1] - wave[i])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', optimize.OptimizeWarning)
                p, _ = optimize.curve_fit(lambda x, a, mu, s: _gaussian(x, a, mu, s, cont), x, y,
                                          p0=[cont - flux[i], 0.0, s0], maxfev=2000)
            if p[0] > 0:
                w, f = x0 + p[1], cont - p[0]
        except (RuntimeError, ValueError):
            pass
    else:
        raise ValueError("center: 'pixel', 'parabola' or 'gaussian'")
    if not (wave[lo] <= w <= wave[hi]):
        return wave[i], flux[i], False, hi - lo + 1
    return w, f, True, hi - lo + 1


# ---------------------------------------------------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------------------------------------------------
def detect_lines(wave, flux, prominence=0.05, center='gaussian', core=CORE, edges='first', error=None, sg_window=9,
                 sg_order=3, edge_sigma=3.0):
    """Absorption lines of a flattened spectrum: every local minimum with this prominence (scipy find_peaks on -flux),
    its maximum on each side (the window of the line), its depth (mean of the two sides minus the minimum) and its
    minimum refined by a fit of the core (center 'gaussian' or 'parabola'; 'pixel': the minimum pixel).

    edges 'first': the first local maximum of the raw flux on each side (one noise pixel can stop it).
    edges 'curvature': window_edges (smoothed first and second derivatives, noise threshold edge_sigma, error = the
    1-sigma flux error, an array or a number; sg_window, sg_order: Savitzky-Golay filter).
    Returns a DataFrame, one row per line: wave_min, flux_min, depth, prominence, i_left, i_right, wave_left_max,
    wave_right_max, flux_left_max, flux_right_max, wave_min_pixel, center_ok, n_core."""
    minima, props = find_peaks(-flux, prominence=prominence)
    n = len(flux)
    if edges == 'curvature':
        if error is None:
            raise ValueError("edges='curvature' needs the flux error (array or number)")
        left, right = window_edges(wave, flux, minima, error, sg_window, sg_order, edge_sigma)
    elif edges == 'first':
        left, right = [], []
        for i in minima:
            il = i - 1
            while il > 0 and flux[il - 1] > flux[il]:
                il -= 1
            ir = i + 1
            while ir < n - 1 and flux[ir + 1] > flux[ir]:
                ir += 1
            left.append(il)
            right.append(ir)
    else:
        raise ValueError("edges: 'first' or 'curvature'")
    rows = []
    for i, prom, il, ir in zip(minima, props['prominences'], left, right):
        if il <= 0 or ir >= n - 1:
            continue
        w_min, f_min, ok, n_core = wave[i], flux[i], True, 1
        if center != 'pixel':
            w_min, f_min, ok, n_core = refine_minimum(wave, flux, i, il, ir, center, core)
        rows.append(dict(wave_min=w_min, flux_min=f_min, depth=0.5 * (flux[il] + flux[ir]) - f_min, prominence=prom,
                         i_left=il, i_right=ir, wave_left_max=wave[il], wave_right_max=wave[ir],
                         flux_left_max=flux[il], flux_right_max=flux[ir], wave_min_pixel=wave[i], center_ok=ok,
                         n_core=n_core))
    return pd.DataFrame(rows)


def window_edges(wave, flux, minima, error, sg_window=9, sg_order=3, edge_sigma=3.0):
    """Window of every line (minimum pixel in `minima`) from the smoothed first and second derivatives, so that noise
    neither stops it early nor carries it across the continuum.

    The flux, its first and its second derivative are smoothed with a Savitzky-Golay filter (sg_window pixels, a
    polynomial of order sg_order); their noise is propagated from `error` through the filter coefficients (white noise
    of the local error, never below the median error: in saturated cores the error is tiny but the flattening and
    the noise of the stack are not). Walking outwards from the minimum:
      1. go on while the smoothed first derivative is significantly rising outwards (> edge_sigma x its noise);
      2. where it is not any more, the candidate edge. It is accepted at once if it is a true maximum (the smoothed
         first derivative has changed sign) and the smoothed second derivative is negative by more than edge_sigma x
         its noise (a curved top between two lines);
      3. otherwise look further out, over sg_window pixels: if the smoothed flux rises again by more than edge_sigma x
         its noise before it drops by as much, the flattening was noise on the wing, and the walk goes on from there
         (1); if it drops first (the next line) or stays flat (continuum), the candidate is the edge.
    The edge is the pixel of the highest raw flux within sg_window // 2 of the candidate (on its side of the minimum).
    Returns (i_left, i_right), arrays of pixel indices."""
    from scipy.signal import savgol_filter, savgol_coeffs
    flux = np.asarray(flux, dtype=np.float64)
    n = len(flux)
    sig = np.broadcast_to(np.asarray(error, dtype=np.float64), flux.shape)
    sig = np.maximum(sig, np.median(sig))      # floor: the error is tiny in saturated cores, not the real scatter
    fs = savgol_filter(flux, sg_window, sg_order)
    d1 = savgol_filter(flux, sg_window, sg_order, deriv=1)          # per pixel
    d2 = savgol_filter(flux, sg_window, sg_order, deriv=2)
    s_f = sig * np.sqrt(np.sum(savgol_coeffs(sg_window, sg_order) ** 2))
    s_d1 = sig * np.sqrt(np.sum(savgol_coeffs(sg_window, sg_order, deriv=1) ** 2))
    s_d2 = sig * np.sqrt(np.sum(savgol_coeffs(sg_window, sg_order, deriv=2) ** 2))
    half = sg_window // 2
    ks = edge_sigma

    def walk(i, s):
        j = i
        while True:
            j += s
            while 0 < j < n - 1 and s * d1[j] > ks * s_d1[j]:      # 1. significantly rising
                j += s
            if j <= 0 or j >= n - 1:
                return min(max(j, 0), n - 1)
            cand = j
            if s * d1[cand] <= 0 and d2[cand] < -ks * s_d2[cand]:  # 2. curved top (a true maximum)
                break
            k, again = cand, False                                 # 3. look further out (sg_window pixels)
            while 0 < k < n - 1 and abs(k - cand) < sg_window:
                k += s
                if fs[k] > fs[cand] + ks * s_f[cand]:
                    again = True
                    break
                if fs[k] < fs[cand] - ks * s_f[cand]:
                    break
            if not again:
                break
            j = k - s
        # highest raw flux near the candidate, on the same side of the minimum
        lo, hi = (max(i + 1, cand - half), min(n - 1, cand + half)) if s > 0 else (max(0, cand - half), min(i - 1, cand + half))
        return lo + int(np.argmax(flux[lo:hi + 1]))

    left = np.array([walk(int(i), -1) for i in minima], dtype=int)
    right = np.array([walk(int(i), 1) for i in minima], dtype=int)
    return left, right


# ---------------------------------------------------------------------------------------------------------------------
# line list
# ---------------------------------------------------------------------------------------------------------------------
def load_vald(path):
    """VALD line list (csv with at least: wave [A, air], depth_vald, species)."""
    return pd.read_csv(path).sort_values('wave', kind='stable').reset_index(drop=True)


def vald_score(dwave, vald_depth, wave, depth, wave_left=None, wave_right=None, score='resolution', resolution=None,
               log_depth_weight=1.0, vald_depth_scale=1.0, distance_weight=0.5, depth_weight=0.5):
    """Score of VALD candidates (dwave = VALD - line minimum [A], their VALD depths) for a line of minimum `wave` and
    `depth`; the identified candidate has the lowest score.
    'resolution': |dwave| / (wave / resolution) + log_depth_weight |ln(vald_depth_scale x VALD depth / depth)|
                  (distance in resolution elements; one resolution element costs as much as a factor e in depth).
    'window'    : distance_weight |dwave| / window width + depth_weight |VALD depth - depth| / depth (before 2026-11-18;
                  the depth term is not bounded and pulled the choice to weaker, more distant VALD lines)."""
    dwave, vald_depth = np.asarray(dwave, dtype=float), np.asarray(vald_depth, dtype=float)
    if score == 'resolution':
        if not resolution:
            raise ValueError("score='resolution' needs the resolution of the spectrum")
        ratio = np.maximum(vald_depth_scale * vald_depth, 1e-3) / max(depth, 1e-3)
        return np.abs(dwave) / (wave / resolution) + log_depth_weight * np.abs(np.log(ratio))
    if score == 'window':
        return (distance_weight * np.abs(dwave) / (wave_right - wave_left)
                + depth_weight * np.abs(vald_depth - depth) / max(depth, 1e-12))
    raise ValueError("score: 'resolution' or 'window'")


def identify(lines, vald, distance_weight=0.5, depth_weight=0.5, min_vald_depth=0.0, blend_depth=0.5, blend_min_sep=0.01,
             score='resolution', resolution=None, log_depth_weight=1.0, vald_depth_scale=None):
    """VALD identification of lines (one row per line with wave_min, depth, wave_left_max, wave_right_max): the
    candidates are the VALD lines between the two maxima of the line (VALD depth >= min_vald_depth); the identified one
    has the lowest vald_score:
    score 'resolution' (default): distance to the minimum in resolution elements (wave / resolution) plus
    log_depth_weight |ln(vald_depth_scale x VALD depth / depth)|. VALD depths are unbroadened; vald_depth_scale None:
    the median over the lines of depth / VALD depth of the closest candidate (stored in out.attrs).
    score 'window': distance_weight |dlambda| / window width + depth_weight |ddepth| / depth (old score).
    The number of candidates is a blend indicator: several VALD lines share the window. Adds vald_index (row of vald,
    -1 if none), vald_wave, vald_depth, species, n_candidates, n_candidates_deep (candidates deeper than blend_depth x
    the line depth), n_blend (those of them more than blend_min_sep [A] from the identified VALD line: neighbours that
    pull the minimum and skew the line; VALD entries at the same wavelength, e.g. hyperfine or isotopic components, are
    not counted), candidates (list)."""
    vw, vd = vald['wave'].values, vald['depth_vald'].values
    if np.any(np.diff(vw) < 0):
        raise ValueError('the line list must be sorted by wavelength (load_vald sorts it)')
    if score == 'resolution' and not resolution:
        raise ValueError("score='resolution' needs the resolution of the spectrum (or score='window', the old score)")
    species = vald['species'].values if 'species' in vald else np.full(len(vald), '')
    keep = vd >= min_vald_depth
    out = lines.copy()
    cols = {k: [] for k in ('vald_index', 'vald_wave', 'vald_depth', 'species', 'n_candidates', 'n_candidates_deep',
                            'n_blend', 'candidates')}
    lo_all = np.searchsorted(vw, out['wave_left_max'].values, side='right')
    hi_all = np.searchsorted(vw, out['wave_right_max'].values, side='left')
    rows = out[['wave_min', 'depth', 'wave_left_max', 'wave_right_max']].values
    cands = []
    for lo, hi in zip(lo_all, hi_all):
        cand = np.arange(lo, hi)
        cands.append(cand[keep[cand]])
    if score == 'resolution' and vald_depth_scale is None:
        r = [d / vd[c[np.argmin(np.abs(vw[c] - w))]] for (w, d, _, _), c in zip(rows, cands)
             if len(c) and vd[c[np.argmin(np.abs(vw[c] - w))]] > 0]
        vald_depth_scale = float(np.median(r)) if r else 1.0
    out.attrs['vald_depth_scale'] = vald_depth_scale
    for (w, d, wl, wr), cand in zip(rows, cands):
        cols['candidates'].append(list(cand))
        cols['n_candidates'].append(len(cand))
        deep = cand[vd[cand] > blend_depth * d]
        cols['n_candidates_deep'].append(len(deep))
        if len(cand):
            s = vald_score(vw[cand] - w, vd[cand], w, d, wl, wr, score, resolution, log_depth_weight, vald_depth_scale,
                           distance_weight, depth_weight)
            j = cand[np.argmin(s)]
            cols['vald_index'].append(j)
            cols['vald_wave'].append(vw[j])
            cols['vald_depth'].append(vd[j])
            cols['species'].append(species[j])
            cols['n_blend'].append(int(np.sum(np.abs(vw[deep] - vw[j]) > blend_min_sep)))
        else:
            cols['n_blend'].append(0)
            cols['vald_index'].append(-1)
            cols['vald_wave'].append(np.nan)
            cols['vald_depth'].append(np.nan)
            cols['species'].append('')
    for k, v in cols.items():
        out[k] = v
    return out
