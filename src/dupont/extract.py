"""A CCF mask from any number of spectra (usually one or two), and its comparison with other masks.

One spectrum: its own lines. Two or more: the lines found in every spectrum (e.g. an observation and a simulation, two
simulations, two instruments). The first spectrum (or `primary`) gives the wavelengths and the weights of the mask
and the frame of the line list and the telluric model; the others only decide which lines are kept and give their own
minima and weights in the table. Every threshold is a flag of the line table, so masks with other choices can be made
afterwards (MaskResult.mask) without running the extraction again.

    spectra = {'PoET': (w, f, e), 'MURaM': (w, f2)}      # flattened (spectra.flatten), same medium, near rest
    res = extract_mask(spectra, ExtractConfig(prominence=0.05), vald=vald, telluric=(w_t, t))
    res.masks['clean']                                   # DataFrame (wave, weight), weights summed to 1
    comp = compare_masks(res.masks['clean'], read_mask('ESPRESSO.csv', power=2))
    table, curves = compare_ccfs({'PoET': (w, f)}, {'own': res.masks['clean'], 'ESPRESSO': ref})
"""
import dataclasses
import json
import os
import re

import numpy as np
import pandas as pd

from . import flags as _flags
from . import lines as _lines
from . import pairing as _pairing
from . import tellurics as _tellurics
from .ccf import RV_GRID, ccf, ccf_params
from .spectra import C_LIGHT

FLAGS = ('telluric', 'shallow', 'no_vald', 'blend')     # every flag: True = the line fails that test


@dataclasses.dataclass
class ExtractConfig:
    """Parameters of extract_mask (saved with every mask)."""
    # detection (lines.detect_lines)
    prominence: float = 0.05                # find_peaks on -flux
    center: str = 'gaussian'                # minimum of a line: 'gaussian', 'parabola' or 'pixel'
    core: float = 0.2                       # core of the line fitted for its minimum (flux < min + core x depth)
    edges: str = 'first'                    # window of a line: 'first' maxima, or 'curvature' (needs the errors)
    edge_sigma: float = 3.0                 # noise threshold of the 'curvature' edges
    # lines common to the spectra (two or more)
    max_dv: float = 2000.0                  # [m/s] pairing of the minima with the primary one, after the global shift
    shift: float = None                     # [m/s] global shift of every other spectrum; None: median of the pairs
    window: str = 'min'                     # window of the weights: 'min' shared (intersection), 'max' union, 'own'
    position: str = 'primary'               # mask wavelength: 'primary' minimum, or 'mean' of the minima (shift removed)
    # weights: Bouchy et al. 2001 eq. 8 with A0 = the flattened flux of each spectrum
    detector_noise: float = 0.0             # sigma_D, in units of the continuum
    # flags
    min_depth: float = 0.0                  # 'shallow': depth below this in one of the spectra
    vald_score: str = 'resolution'          # line-list identification score (lines.vald_score): 'resolution' or 'window'
    log_depth_weight: float = 1.0           # 'resolution': weight of |ln(scale x VALD depth / depth)|
    vald_depth_scale: float = None          # 'resolution': VALD depth scale; None = median depth / VALD depth
    distance_weight: float = 0.5            # 'window' score only (the old one)
    depth_weight: float = 0.5
    blend_depth: float = 0.5                # 'blend': another line-list line deeper than blend_depth x the line depth in
    blend_min_sep: float = 0.01             # the window, more than blend_min_sep [A] from the identified one
    require_vald: bool = False              # 'no_vald' removes the line from the default masks
    telluric_threshold: float = 0.01        # 'telluric': telluric depth in the window above this
    telluric_dv: tuple = (-1000.0, 1000.0)  # [m/s] range of telluric velocities relative to the lines
    resolution: float = None                # [-] resolution of the spectra (distance unit of the 'resolution' VALD score)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MaskResult:
    """Output of extract_mask.

    lines  : one row per line: wave, weight (the mask), <name>_<column> of every spectrum (wave_min, depth, window,
             weight, sigma_rv_photon if the spectrum has errors), dv_<name> of the other spectra, the line-list
             identification (vald_*), telluric_depth and the flags (FLAGS);
    masks  : 'all' (no telluric, no shallow line, and no line without a line-list line if require_vald) and 'clean'
             (also no blend), DataFrames (wave, weight) with weights summed to 1;
    counts : number of lines at every step; shifts: {name: global shift of that spectrum - primary [m/s]}."""
    lines: pd.DataFrame
    masks: dict
    counts: dict
    shifts: dict
    names: list
    config: dict

    @property
    def primary(self):
        return self.names[0]

    def mask(self, exclude=('telluric', 'shallow'), weight=None, position='wave', lines=None):
        """Mask (wave, weight) of the lines with none of the flags in `exclude`. weight: name of the spectrum whose
        weights are used (default: primary), or a column of lines; position: column of the wavelengths ('wave', or
        '<name>_wave_min'). lines: boolean selection applied on top (e.g. lines['wave'] > 5000)."""
        t = self.lines
        sel = _flags.select(t, [f for f in exclude if f in t]) if exclude else pd.Series(True, index=t.index)
        if lines is not None:
            sel &= lines
        col = 'weight' if weight is None else (weight if weight in t else '%s_weight' % _label(weight))
        return _norm(t.loc[sel, position].values, t.loc[sel, col].values)

    def flag_summary(self, flags=FLAGS):
        """Lines and share of the weight of every flag (flags.summary)."""
        return _flags.summary(self.lines, [f for f in flags if f in self.lines], weight='weight')

    def save(self, folder, tag='mask'):
        """Every mask as <folder>/<tag>_<name>.csv (wave, weight), the line table as <tag>_lines.csv, the parameters,
        names, shifts and counts as <tag>_config.json."""
        os.makedirs(folder, exist_ok=True)
        for name, m in self.masks.items():
            write_mask(m, os.path.join(folder, '%s_%s.csv' % (tag, name.replace(' ', '_'))))
        self.lines.drop(columns=['vald_candidates'], errors='ignore').to_csv(os.path.join(folder, '%s_lines.csv' % tag),
                                                                            index=False)
        meta = {'names': self.names, 'config': self.config, 'counts': self.counts, 'shifts_m_s': self.shifts}
        with open(os.path.join(folder, '%s_config.json' % tag), 'w') as f:
            json.dump(meta, f, indent=1, default=float)


# ---------------------------------------------------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------------------------------------------------
def _label(name):
    """Column prefix of a spectrum name."""
    return re.sub(r'\W+', '_', str(name)).strip('_')


def _as_dict(spectra, primary):
    if isinstance(spectra, (tuple, list)) and len(spectra) in (2, 3) and np.ndim(spectra[0]) == 1:
        spectra = {'spectrum': spectra}
    spectra = dict(spectra)
    if not 1 <= len(spectra):
        raise ValueError('at least one spectrum')
    primary = list(spectra)[0] if primary is None else primary
    if primary not in spectra:
        raise KeyError('primary %r is not one of the spectra %s' % (primary, list(spectra)))
    names = [primary] + [n for n in spectra if n != primary]
    if len({_label(n) for n in names}) != len(names):
        raise ValueError('two spectrum names give the same column prefix: %s' % names)
    out = {}
    for n in names:
        s = spectra[n]
        w, f = (np.asarray(a, dtype=np.float64) for a in s[:2])
        e = None if len(s) < 3 or s[2] is None else np.asarray(s[2], dtype=np.float64)
        o = np.argsort(w)
        out[n] = (w[o], f[o], None if e is None else e[o])
    return out, names


def common_grid_many(spectra, names):
    """All the spectra on the wavelengths of the first one (names[0]) in the range common to all; the others
    interpolated linearly (unless already on the same grid); pixels not finite in one spectrum removed from all.
    Returns (wave, {name: flux}, {name: error or None})."""
    lo = max(spectra[n][0][0] for n in names)
    hi = min(spectra[n][0][-1] for n in names)
    w0 = spectra[names[0]][0]
    k = (w0 >= lo) & (w0 <= hi)
    wave = w0[k]
    flux, err = {}, {}
    for n in names:
        w, f, e = spectra[n]
        if len(w) == len(w0) and np.array_equal(w, w0):
            flux[n], err[n] = f[k], None if e is None else e[k]
        else:
            ok = np.isfinite(f)
            flux[n] = np.interp(wave, w[ok], f[ok], left=np.nan, right=np.nan)
            err[n] = None if e is None else np.interp(wave, w[ok], e[ok], left=np.nan, right=np.nan)
    good = np.all([np.isfinite(flux[n]) for n in names], axis=0)
    return wave[good], {n: flux[n][good] for n in names}, {n: None if err[n] is None else err[n][good] for n in names}


def extract_mask(spectra, config=None, vald=None, telluric=None, primary=None):
    """Mask of the lines found in every spectrum.

    spectra : {name: (wave, flux) or (wave, flux, error)} (one or more), or a single (wave, flux[, error]) tuple. The
              spectra must be flattened (continuum 1), in the same wavelength medium and at the same resolution (or
              on purpose not: e.g. a synthetic spectrum at its native resolution), close to the same velocity frame
              (a global shift up to ~3 max_dv is measured and removed for the pairing). The others are interpolated on
              the wavelengths of the primary one, in the range common to all.
    config  : ExtractConfig.
    vald    : line list (lines.load_vald: wave, depth_vald, species), optional: identification and the flags
              'no_vald' and 'blend'.
    telluric: (wave, transmission) of a telluric model in the frame of the primary spectrum, optional: flag 'telluric'.
    primary : name of the spectrum that gives the mask wavelengths and weights (default: the first).
    Returns a MaskResult."""
    cfg = config or ExtractConfig()
    spectra, names = _as_dict(spectra, primary)
    wave, flux, err = common_grid_many(spectra, names)
    counts = {'spectra': len(names), 'pixels (common grid)': len(wave)}

    # 1. lines of every spectrum
    found = {}
    for n in names:
        if cfg.edges == 'curvature' and err[n] is None:
            raise ValueError("edges='curvature' needs the error of %r" % n)
        found[n] = _lines.detect_lines(wave, flux[n], cfg.prominence, cfg.center, cfg.core, edges=cfg.edges,
                                       error=err[n], edge_sigma=cfg.edge_sigma)
        counts['%s lines detected' % n] = len(found[n])
    p = names[0]
    base = found[p].sort_values('wave_min').reset_index(drop=True)
    base['_id'] = np.arange(len(base))

    # 2. lines in every spectrum: each other spectrum paired one to one with the primary one
    rows, shifts, keep = {p: base.set_index('_id')}, {}, set(base['_id'])
    for n in names[1:]:
        a, b, shift = _pairing.pair_lines(base, found[n], max_dv=cfg.max_dv, shift=cfg.shift)
        b.index = a['_id'].values
        rows[n], shifts[n] = b.drop(columns=['_id'], errors='ignore'), shift
        keep &= set(a['_id'])
        counts['%s lines paired with %s' % (n, p)] = len(a)
    ids = np.array(sorted(keep), dtype=int)
    tabs = {n: rows[n].loc[ids].reset_index(drop=True) for n in names}
    counts['lines in every spectrum'] = len(ids)

    # 3. window of the weights
    tabs = _windows(tabs, names, wave, cfg.window)
    counts['lines with a window'] = len(tabs[p])

    # 4. weights (Bouchy eq. 8, A0 = flux) and photon-noise RV error of the spectra with errors
    for n in names:
        t = _pairing.rv_weights(wave, flux[n], None, tabs[n], cfg.detector_noise)
        if err[n] is not None:
            t['sigma_rv_photon'] = _pairing.rv_weights(wave, flux[n], err[n], tabs[n])['sigma_rv'].values
        tabs[n] = t.drop(columns=['sigma_rv', 'dv'], errors='ignore')

    # 5. one table
    t = pd.concat([tabs[n].add_prefix(_label(n) + '_') for n in names], axis=1)
    lp = _label(p)
    if cfg.position == 'mean' and len(names) > 1:
        ws = [tabs[p]['wave_min'].values] + [tabs[n]['wave_min'].values * (1 - shifts[n] / C_LIGHT) for n in names[1:]]
        t.insert(0, 'wave', np.mean(ws, axis=0))
    elif cfg.position in ('primary', 'mean'):
        t.insert(0, 'wave', t[lp + '_wave_min'].values)
    else:
        raise ValueError("position: 'primary' or 'mean'")
    t.insert(1, 'weight', t[lp + '_weight'].values)
    for n in names[1:]:
        t['dv_' + _label(n)] = C_LIGHT * (t[_label(n) + '_wave_min'] - t[lp + '_wave_min']) / t[lp + '_wave_min'] - shifts[n]

    # 6. flags
    t['shallow'] = np.min([t[_label(n) + '_depth'].values for n in names], axis=0) < cfg.min_depth
    if vald is not None:
        q = pd.DataFrame({'wave_min': t['wave'], 'depth': t[lp + '_depth'], 'wave_left_max': t[lp + '_wave_left_max'],
                          'wave_right_max': t[lp + '_wave_right_max']})
        q = _lines.identify(q, vald, cfg.distance_weight, cfg.depth_weight, blend_depth=cfg.blend_depth,
                            blend_min_sep=cfg.blend_min_sep, score=cfg.vald_score, resolution=cfg.resolution,
                            log_depth_weight=cfg.log_depth_weight, vald_depth_scale=cfg.vald_depth_scale)
        for c in ('vald_index', 'vald_wave', 'vald_depth', 'species', 'n_candidates', 'n_candidates_deep', 'n_blend'):
            t[c if c.startswith('vald') else 'vald_' + c] = q[c].values
        t['vald_candidates'] = q['candidates'].values
        t['no_vald'] = t['vald_index'].values < 0
        t['blend'] = t['vald_n_blend'].values > 0
    else:
        t['no_vald'], t['blend'] = False, False
    if telluric is not None:
        t['telluric_depth'] = _tellurics.telluric_depth(t[lp + '_wave_left_max'].values, t[lp + '_wave_right_max'].values,
                                                        np.asarray(telluric[0]), np.asarray(telluric[1]), cfg.telluric_dv)
        t['telluric'] = t['telluric_depth'].values > cfg.telluric_threshold
    else:
        t['telluric_depth'], t['telluric'] = np.nan, False
    for f in FLAGS:
        counts['flag %s' % f] = int(t[f].sum())

    base_flags = ['telluric', 'shallow'] + (['no_vald'] if cfg.require_vald else [])
    res = MaskResult(lines=t, masks={}, counts=counts, shifts=shifts, names=names, config=cfg.to_dict())
    res.masks['all'] = res.mask(base_flags)
    res.masks['clean'] = res.mask(base_flags + ['blend'])
    counts['mask all'], counts['mask clean'] = len(res.masks['all']), len(res.masks['clean'])
    return res


def _windows(tabs, names, wave, how):
    """The same window (pixels) for every spectrum of a line: 'min' intersection, 'max' union, 'own' unchanged. The
    first windows are kept in wave_left_max_own, wave_right_max_own. Lines with an empty window are removed."""
    if how == 'own' or len(names) == 1:
        return tabs
    if how not in ('min', 'max'):
        raise ValueError("window: 'min', 'max' or 'own'")
    L = np.array([tabs[n]['i_left'].values for n in names])
    R = np.array([tabs[n]['i_right'].values for n in names])
    il, ir = (L.max(0), R.min(0)) if how == 'min' else (L.min(0), R.max(0))
    ok = ir > il
    out = {}
    for n in names:
        t = tabs[n][ok].reset_index(drop=True).copy()
        t['wave_left_max_own'], t['wave_right_max_own'] = t['wave_left_max'], t['wave_right_max']
        t['i_left'], t['i_right'] = il[ok], ir[ok]
        t['wave_left_max'], t['wave_right_max'] = wave[il[ok]], wave[ir[ok]]
        out[n] = t
    return out


def _norm(w, f):
    w, f = np.asarray(w, dtype=np.float64), np.asarray(f, dtype=np.float64)
    o = np.argsort(w)
    return pd.DataFrame({'wave': w[o], 'weight': f[o] / np.sum(f)})


# ---------------------------------------------------------------------------------------------------------------------
# masks on disk
# ---------------------------------------------------------------------------------------------------------------------
def read_mask(path, wave=None, weight=None, power=1.0, air=True):
    """A mask as a DataFrame (wave [A], weight) with the weights summed to 1, from
      - a csv: wavelength column `wave` (default: wave, lambda, wavelength) and weight column `weight` (default: weight,
        contrast, cont, depth; none: every line weighs the same);
      - a FITS table (e.g. ESPRESSO DRS masks: lambda, contrast), the same column names;
      - a whitespace text file (.mas, .txt, .dat): 3 columns (start, end, weight; HARPS masks: wave = middle) or 2
        (wave, weight).
    weight = column ** power (power 2: contrast^2, as for the ESPRESSO mask).
    air False: the wavelengths are in vacuum and are converted to air (spectra.vacuum_to_air)."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.fits', '.fit'):
        from astropy.table import Table
        d = Table.read(path).to_pandas()
    elif ext in ('.mas', '.txt', '.dat'):
        a = np.loadtxt(path, comments='#')
        d = (pd.DataFrame({'wave': 0.5 * (a[:, 0] + a[:, 1]), 'weight': a[:, 2]}) if a.shape[1] >= 3 else
             pd.DataFrame({'wave': a[:, 0], 'weight': a[:, 1]}))
    else:
        d = pd.read_csv(path)
    cols = {c.lower(): c for c in d.columns}

    def pick(given, options, what):
        if given is not None:
            return given
        for o in options:
            if o in cols:
                return cols[o]
        if what == 'weight':
            return None
        raise KeyError('no %s column in %s (columns %s)' % (what, path, list(d.columns)))
    wc = pick(wave, ('wave', 'lambda', 'wavelength', 'wave_air', 'll'), 'wavelength')
    fc = pick(weight, ('weight', 'contrast', 'cont', 'depth'), 'weight')
    w = d[wc].values.astype(np.float64)
    f = np.ones(len(w)) if fc is None else np.abs(d[fc].values.astype(np.float64)) ** power
    if not air:
        from .spectra import vacuum_to_air
        w = vacuum_to_air(w)
    return _norm(w, f)


def write_mask(mask, path):
    """A mask (wave, weight) as a csv (wave, weight), or as a .mas text file (start, end, weight; start = end = wave,
    as the .mas masks read by SunSim)."""
    path = str(path)
    if path.lower().endswith('.mas'):
        np.savetxt(path, np.column_stack([mask['wave'], mask['wave'], mask['weight']]), fmt=['%.6f', '%.6f', '%.6e'],
                   delimiter='\t')
    else:
        mask[['wave', 'weight']].to_csv(path, index=False)


# ---------------------------------------------------------------------------------------------------------------------
# comparison with other masks
# ---------------------------------------------------------------------------------------------------------------------
def compare_masks(mask, reference, max_dv=3000.0, shift=0.0, names=('mask', 'reference')):
    """One-to-one match of two masks (DataFrames wave, weight): pairs within max_dv [m/s], the closest first, after a
    global shift of the reference (shift [m/s]: 0 by default, both masks at rest; None: the median of the pairs within
    3 max_dv, for masks in different frames).

    Returns a dict:
      'matched': one row per pair (wave, weight, ref_wave, ref_weight, dv = mask - reference [m/s], the global shift
                 removed), the
                 weights normalised over the matched lines;
      'counts' : lines of each mask, matched lines, share of the weight of each mask in the matched lines, median and
                 90th percentile of |dv|, the correlation of the matched weights (Pearson r of the weights and of their
                 logarithms);
      'masks'  : '<mask> common' and '<reference> common' (the matched lines, each with its own wavelengths and
                 weights), for CCFs on the same lines;
      'shift'  : the shift used [m/s]."""
    a = mask.sort_values('wave').reset_index(drop=True)
    b = reference.sort_values('wave').reset_index(drop=True)
    wa, wb = a['wave'].values, b['wave'].values
    if shift is None:
        first = _pairing._closest_first(wa, wb, 3 * max_dv)
        shift = float(np.median([C_LIGHT * (wb[j] - wa[i]) / wa[i] for i, j in first])) if first else 0.0
    pairs = _pairing._closest_first(wa, wb, max_dv, shift)
    ia = np.array([q[0] for q in pairs], dtype=int)
    ib = np.array([q[1] for q in pairs], dtype=int)
    m = pd.DataFrame({'wave': wa[ia], 'weight': a['weight'].values[ia], 'ref_wave': wb[ib],
                      'ref_weight': b['weight'].values[ib]})
    m['dv'] = C_LIGHT * (m['wave'] - m['ref_wave']) / m['wave'] + shift
    share_a, share_b = m['weight'].sum() / a['weight'].sum(), m['ref_weight'].sum() / b['weight'].sum()
    m['weight'] /= m['weight'].sum()
    m['ref_weight'] /= m['ref_weight'].sum()
    pos = (m['weight'] > 0) & (m['ref_weight'] > 0)
    counts = {'%s lines' % names[0]: len(a), '%s lines' % names[1]: len(b), 'matched lines': len(m),
              '%s weight matched [%%]' % names[0]: 100 * share_a, '%s weight matched [%%]' % names[1]: 100 * share_b,
              'median |dv| [m/s]': float(np.median(np.abs(m['dv']))) if len(m) else np.nan,
              '90% |dv| [m/s]': float(np.percentile(np.abs(m['dv']), 90)) if len(m) else np.nan,
              'weight r': float(np.corrcoef(m['weight'], m['ref_weight'])[0, 1]) if len(m) > 2 else np.nan,
              'log weight r': float(np.corrcoef(np.log(m.loc[pos, 'weight']), np.log(m.loc[pos, 'ref_weight']))[0, 1])
              if pos.sum() > 2 else np.nan}
    masks = {'%s common' % names[0]: _norm(m['wave'], m['weight']),
             '%s common' % names[1]: _norm(m['ref_wave'], m['ref_weight'])}
    return {'matched': m, 'counts': counts, 'masks': masks, 'shift': shift}


def compare_ccfs(spectra, masks, rv=RV_GRID):
    """CCF of every spectrum with every mask, and its RV, FWHM, contrast and BIS (ccf_params).
    spectra: {name: (wave, flux[, error])} flattened; masks: {name: DataFrame (wave, weight)}.
    Returns (table indexed by (spectrum, mask) with rv, fwhm, contrast, bis [m/s] and the number of mask lines inside
    the spectrum, {(spectrum, mask): (ccf, params)} for plots.ccfs_and_bisectors)."""
    rows, curves = [], {}
    for sn, s in spectra.items():
        w, f = np.asarray(s[0], dtype=np.float64), np.asarray(s[1], dtype=np.float64)
        ok = np.isfinite(f)
        w, f = w[ok], f[ok]
        for mn, m in masks.items():
            c = ccf(w, f, m['wave'].values, m['weight'].values, rv)
            q = ccf_params(rv, c)
            curves[(sn, mn)] = (c, q)
            n_in = int(np.sum((m['wave'].values > w[0]) & (m['wave'].values < w[-1])))
            rows.append({'spectrum': sn, 'mask': mn, 'rv': q['rv'], 'fwhm': q['fwhm'], 'contrast': q['contrast'],
                         'bis': q['bis'], 'lines used': n_in})
    return pd.DataFrame(rows).set_index(['spectrum', 'mask']), curves
