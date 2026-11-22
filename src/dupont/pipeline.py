"""The whole mask construction: lines common to an observed and a synthetic spectrum, identified in a line list,
weighted by their RV information, without telluric lines, cross-matched with a reference mask."""
import dataclasses
import json
import os

import numpy as np
import pandas as pd

from . import lines as _lines
from . import pairing as _pairing
from . import tellurics as _tellurics
from .spectra import C_LIGHT, common_grid


@dataclasses.dataclass
class MaskConfig:
    """Parameters of build_mask (saved with every mask)."""
    names: tuple = ('PoET', 'MURaM')        # observed, synthetic
    prominence: float = 0.05                # line detection (find_peaks on -flux)
    center: str = 'gaussian'                # minimum of a line: 'gaussian', 'parabola' or 'pixel'
    core: float = 0.2                       # core of the line fitted for its minimum
    max_dv: float = 2000.0                  # [m/s] pairing of the minima of the two spectra (after the global shift)
    window: str = 'min'                     # window of the weight: 'min' (shared, intersection), 'max', 'own'
    vald_score: str = 'resolution'          # VALD identification score (lines.vald_score): 'resolution' or 'window' (old)
    log_depth_weight: float = 1.0           # 'resolution': weight of |ln(scale x VALD depth / depth)|
    vald_depth_scale: float = None          # 'resolution': VALD depth scale; None = median depth / VALD depth
    distance_weight: float = 0.5            # 'window' score only
    depth_weight: float = 0.5
    require_vald: bool = False              # drop the common lines with no VALD line in their window
    blend_depth: float = 0.5                # blend: another VALD line deeper than blend_depth x the line depth in its
    blend_min_sep: float = 0.01             # window, more than blend_min_sep [A] from the identified VALD line
    telluric_threshold: float = 0.01        # telluric depth in the window above which a line is telluric
    telluric_dv: tuple = (-1000.0, 1000.0)  # [m/s] range of telluric velocities relative to the lines
    ref_tolerance: float = 0.11             # [A] match of the reference mask to the lines (VALD wavelength)
    ref_max_dwave: float = 0.05             # [A] both minima within this of the reference line to keep it
    detector_noise: float = 0.0             # sigma_D of Bouchy eq. 8, in units of the continuum (weights)
    resolution: float = None                # [-] resolution the synthetic spectrum was degraded to (distance unit of
                                            # the 'resolution' VALD score)

    def to_dict(self):
        return dataclasses.asdict(self)


def build_mask(observed, synthetic, vald, config=None, reference=None, telluric=None):
    """Lines and masks.

    observed: (wave, flux, error) flattened, at rest (e.g. the PoET stack, shifted so its CCF with the reference mask
    is at 0); synthetic: (wave, flux) flattened, at the resolution of the observation, at rest; vald: line list
    (lines.load_vald); reference: DataFrame (wave, weight) of a reference mask (e.g. ESPRESSO), optional; telluric:
    (wave, transmission) of the telluric model in the frame of the observed spectrum, optional.

    Returns a dict:
      'lines': one row per common line (columns of both spectra prefixed obs_ / syn_, VALD identification, telluric
               depth, reference match, flags 'telluric', 'in_reference', 'selected');
      'masks': {name: DataFrame (wave, weight), weights summed to 1} with
               '<obs> all': common lines, telluric free (and with a VALD line if config.require_vald);
               '<obs> clean': those that are not blends (no other VALD line deeper than config.blend_depth x the
               line depth in the window, more than config.blend_min_sep from the identified one);
               '<obs> common': those also in the reference mask (observed minima and weights);
               'reference common': the same lines with the reference wavelengths and weights;
               'reference all': the whole reference mask;
      'counts': number of lines at every step; 'shift': global velocity shift synthetic - observed [m/s];
      'config': the parameters."""
    cfg = config or MaskConfig()
    obs, syn = cfg.names
    w_o, f_o, e_o = observed
    w_s, f_s = synthetic[:2]
    wave, f_o, f_s, e_o, _ = common_grid(w_o, f_o, w_s, f_s, error_a=e_o)
    counts = {}

    lo = _lines.detect_lines(wave, f_o, cfg.prominence, cfg.center, cfg.core)
    ls = _lines.detect_lines(wave, f_s, cfg.prominence, cfg.center, cfg.core)
    counts['%s lines detected' % obs], counts['%s lines detected' % syn] = len(lo), len(ls)

    co, cs, shift = _pairing.pair_lines(lo, ls, max_dv=cfg.max_dv)
    co, cs = _pairing.shared_window(co, cs, wave, wave, cfg.window)
    counts['common lines'] = len(co)

    co = _lines.identify(co, vald, cfg.distance_weight, cfg.depth_weight, blend_depth=cfg.blend_depth,
                         blend_min_sep=cfg.blend_min_sep, score=cfg.vald_score, resolution=cfg.resolution,
                         log_depth_weight=cfg.log_depth_weight, vald_depth_scale=cfg.vald_depth_scale)
    # mask weights = Bouchy eq. 8 with A0 = flattened flux of each spectrum; sigma_rv_stack = photon-noise RV error of
    # each line in the observed stack (its measured error)
    co = _pairing.rv_weights(wave, f_o, None, co, cfg.detector_noise)
    cs = _pairing.rv_weights(wave, f_s, None, cs, cfg.detector_noise)
    co['sigma_rv_stack'] = _pairing.rv_weights(wave, f_o, e_o, co)['sigma_rv'].values

    t = pd.concat([co.add_prefix('obs_'), cs.add_prefix('syn_')], axis=1)
    for c in ('vald_index', 'vald_wave', 'vald_depth', 'species', 'n_candidates', 'n_candidates_deep', 'n_blend',
              'candidates'):
        t[c] = t.pop('obs_' + c)
    t['dv'] = t.pop('obs_dv')
    t = t.drop(columns=['syn_dv'])
    t['has_vald'] = t['vald_index'] >= 0
    t['blend'] = t['n_blend'] > 0
    counts['common lines with a VALD line'] = int(t['has_vald'].sum())
    counts['blends (a deep VALD neighbour in the window)'] = int(t['blend'].sum())

    if telluric is not None:
        t['telluric_depth'] = _tellurics.telluric_depth(t['obs_wave_left_max'].values, t['obs_wave_right_max'].values,
                                                        telluric[0], telluric[1], cfg.telluric_dv)
        t['telluric'] = t['telluric_depth'] > cfg.telluric_threshold
    else:
        t['telluric_depth'], t['telluric'] = np.nan, False
    counts['telluric lines'] = int(t['telluric'].sum())
    t['selected'] = ~t['telluric'] & (t['has_vald'] | (not cfg.require_vald))
    counts['%s all (selected)' % obs] = int(t['selected'].sum())

    # reference mask: one-to-one match on the VALD wavelength (the mean minimum when there is none)
    t['in_reference'], t['ref_wave'], t['ref_weight'] = False, np.nan, np.nan
    masks = {}
    sel = t[t['selected']]
    masks['%s all' % obs] = _norm(sel['obs_wave_min'].values, sel['obs_weight'].values)
    clean = sel[~sel['blend']]
    masks['%s clean' % obs] = _norm(clean['obs_wave_min'].values, clean['obs_weight'].values)
    counts['%s clean (selected, not blends)' % obs] = len(clean)
    if reference is not None:
        key = np.where(sel['has_vald'], sel['vald_wave'], 0.5 * (sel['obs_wave_min'] + sel['syn_wave_min']))
        m = _pairing.match_reference(key, reference['wave'].values, cfg.ref_tolerance)
        idx = sel.index.values[m['line_index'].values]
        rw = reference['wave'].values[m['ref_index'].values]
        close = ((np.abs(t.loc[idx, 'obs_wave_min'].values - rw) < cfg.ref_max_dwave)
                 & (np.abs(t.loc[idx, 'syn_wave_min'].values - rw) < cfg.ref_max_dwave))
        counts['reference lines matched'] = len(m)
        idx, ri = idx[close], m['ref_index'].values[close]
        t.loc[idx, 'in_reference'] = True
        t.loc[idx, 'ref_wave'] = reference['wave'].values[ri]
        t.loc[idx, 'ref_weight'] = reference['weight'].values[ri]
        counts['reference lines selected'] = len(idx)
        c = t.loc[np.sort(idx)]
        masks['%s common' % obs] = _norm(c['obs_wave_min'].values, c['obs_weight'].values)
        masks['reference common'] = _norm(c['ref_wave'].values, c['ref_weight'].values)
        masks['reference all'] = _norm(reference['wave'].values, reference['weight'].values)
    return {'lines': t, 'masks': masks, 'counts': counts, 'shift': shift, 'config': cfg.to_dict()}


def _norm(w, f):
    o = np.argsort(w)
    return pd.DataFrame({'wave': np.asarray(w, float)[o], 'weight': np.asarray(f, float)[o] / np.sum(f)})


def load_reference_mask(path, weight='contrast2'):
    """Reference mask from a csv with wave and cont (contrast) columns, e.g. the ESPRESSO mask: weight = contrast^2
    ('contrast2') or contrast ('contrast'). Returns a DataFrame (wave, weight)."""
    d = pd.read_csv(path)
    w = d['cont'].values ** 2 if weight == 'contrast2' else d['cont'].values
    return pd.DataFrame({'wave': d['wave'].values, 'weight': w / np.sum(w)})


def save(result, folder, tag='mask'):
    """Every mask as <folder>/<tag>_<name>.csv (wave, weight), the line table as <tag>_lines.csv and the parameters
    and counts as <tag>_config.json."""
    os.makedirs(folder, exist_ok=True)
    for name, m in result['masks'].items():
        m.to_csv(os.path.join(folder, '%s_%s.csv' % (tag, name.replace(' ', '_'))), index=False)
    result['lines'].drop(columns=['candidates']).to_csv(os.path.join(folder, '%s_lines.csv' % tag), index=False)
    meta = {'config': result['config'], 'counts': result['counts'], 'shift_m_s': result['shift']}
    with open(os.path.join(folder, '%s_config.json' % tag), 'w') as f:
        json.dump(meta, f, indent=1, default=float)
