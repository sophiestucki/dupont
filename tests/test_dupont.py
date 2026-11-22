"""Injection-recovery tests on synthetic spectra with known lines."""
import numpy as np
import pandas as pd
import pytest

import dupont as lm

C = lm.C_LIGHT
R_TEST = 190000.0


def _spectrum(centres, depths, sigma_v=3000.0, wave=None, noise=0.0, seed=0):
    """Flattened spectrum with gaussian lines (width sigma_v [m/s]) and optional white noise."""
    if wave is None:
        wave = np.exp(np.arange(np.log(5000.0), np.log(5050.0), 1.0 / 600000))
    f = np.ones_like(wave)
    for c, d in zip(centres, depths):
        s = c * sigma_v / C
        f -= d * np.exp(-0.5 * ((wave - c) / s) ** 2)
    rng = np.random.default_rng(seed)
    e = np.full_like(wave, noise if noise > 0 else 1e-3)
    return wave, f + noise * rng.standard_normal(len(wave)), e


def _lines(n=40, seed=1):
    rng = np.random.default_rng(seed)
    centres = np.sort(rng.uniform(5002, 5048, n))
    centres = centres[np.concatenate([[True], np.diff(centres) > 0.4])]   # isolated lines
    depths = rng.uniform(0.1, 0.8, len(centres))
    return centres, depths


def test_detect_and_minimum():
    c, d = _lines()
    w, f, _ = _spectrum(c, d)
    t = lm.detect_lines(w, f, prominence=0.05)
    assert len(t) == len(c)
    dv = C * (t['wave_min'].values - c) / c
    assert np.max(np.abs(dv)) < 5.0                          # gaussian core fit: m/s precision without noise
    assert np.allclose(t['depth'].values, d, atol=0.02)


def test_degrade_resolution_width():
    c, d = np.array([5025.0]), np.array([0.5])
    w, f, _ = _spectrum(c, d, sigma_v=1000.0)
    R = 190000.0
    g = lm.degrade_resolution(w, f, R)
    s_ip = C / R / (2 * np.sqrt(2 * np.log(2)))
    s_expected = np.hypot(1000.0, s_ip)
    # area conserved, width = quadratic sum
    area0, area1 = np.trapezoid(1 - f, w), np.trapezoid(1 - g, w)
    assert abs(area1 / area0 - 1) < 1e-3
    s_meas = np.sqrt(np.trapezoid((1 - g) * (w - 5025.0) ** 2, w) / area1) / 5025.0 * C
    assert abs(s_meas / s_expected - 1) < 0.01


def test_pairing_with_shift_and_one_to_one():
    c, d = _lines()
    w, f, _ = _spectrum(c, d)
    shift = 300.0
    w2, f2, _ = _spectrum(c * (1 + shift / C), d)
    a, b = lm.detect_lines(w, f), lm.detect_lines(w2, f2)
    ca, cb, s = lm.pair_lines(a, b, max_dv=2000.0)
    assert len(ca) == len(c)
    assert abs(s - shift) < 10
    assert np.max(np.abs(ca['dv'])) < 10


def test_weights_scale_with_noise():
    c, d = _lines()
    w, f, e = _spectrum(c, d)
    t = lm.detect_lines(w, f)
    w1 = lm.rv_weights(w, f, e, t)['weight'].values
    w2 = lm.rv_weights(w, f, 2 * e, t)['weight'].values
    assert np.allclose(w1 / w2, 4.0)
    # deeper lines carry more information (same width)
    assert np.corrcoef(t['depth'], w1)[0, 1] > 0.9



def test_weights_bouchy_eq8_flux_variance():
    c, d = _lines()
    w, f, _ = _spectrum(c, d)
    t = lm.detect_lines(w, f)
    wf = lm.rv_weights(w, f, None, t)['weight'].values
    w1 = lm.rv_weights(w, f, np.ones_like(f), t)['weight'].values
    assert np.all(wf >= w1)
    wd = lm.rv_weights(w, f, None, t, detector_noise=0.1)['weight'].values
    assert np.all(wd < wf)
    assert np.corrcoef(np.log(wf), np.log(w1))[0, 1] > 0.95

def test_match_reference_one_to_one():
    lines = np.array([5000.00, 5001.00])
    ref = np.array([4999.98, 5000.03, 5001.20])          # two reference lines close to the first line
    m = lm.match_reference(lines, ref, max_dwave=0.11)
    assert len(m) == 1 and m['line_index'].tolist() == [0] and m['ref_index'].tolist() == [0]


def test_telluric_depth_window():
    tw = np.linspace(5000, 5010, 10001)
    tt = np.ones_like(tw)
    tt[np.abs(tw - 5005.0) < 0.002] = 0.9                  # one telluric line at 5005 A
    left, right = np.array([5004.95, 5005.03]), np.array([5004.99, 5005.08])
    d0 = lm.telluric_depth(left, right, tw, tt, dv_range=(0.0, 0.0))
    assert np.allclose(d0, 0.0)
    d1 = lm.telluric_depth(left, right, tw, tt, dv_range=(-2000.0, 2000.0))   # +-0.033 A
    assert d1[0] > 0.09 and d1[1] > 0.09


def test_ccf_rv_recovered():
    c, d = _lines()
    v = 1234.0
    w, f, _ = _spectrum(c * (1 + v / C), d)
    rv = lm.RV_GRID
    p = lm.ccf_params(rv, lm.ccf(w, f, c, d))
    assert abs(p['rv'] - v) < 5
    assert abs(p['bis']) < 5                                 # symmetric lines


def test_build_mask_end_to_end():
    c, d = _lines(60, seed=3)
    w, f_obs, e = _spectrum(c, d, noise=0.002, seed=4)
    # synthetic: same lines, one missing, one extra, slightly shifted
    keep = np.ones(len(c), bool)
    keep[5] = False
    g = np.argmax(np.diff(c))
    extra = 0.5 * (c[g] + c[g + 1])                         # in the largest gap: no blend
    cs = np.append(c[keep], extra) * (1 + 150.0 / C)
    ds = np.append(d[keep], 0.5)
    _, f_syn, _ = _spectrum(cs, ds, wave=w)
    vald = pd.DataFrame({'wave': c, 'depth_vald': d, 'species': 'Fe 1'})
    ref = pd.DataFrame({'wave': c[::2], 'weight': d[::2]})
    tw = np.linspace(4990, 5060, 70001)
    tt = np.ones_like(tw)
    tt[np.abs(tw - c[10]) < 0.003] = 0.95                   # a telluric line on top of line 10
    r = lm.build_mask((w, f_obs, e), (w, f_syn), vald, lm.MaskConfig(resolution=R_TEST), reference=ref, telluric=(tw, tt))
    t = r['lines']
    assert abs(r['shift'] - 150.0) < 20
    assert r['counts']['common lines'] == len(c) - 1
    assert t.loc[np.isclose(t['vald_wave'], c[10]), 'telluric'].all()
    assert not np.any(np.isclose(r['masks']['PoET all']['wave'], c[10], atol=0.01))
    assert np.isclose(r['masks']['PoET all']['weight'].sum(), 1.0)
    assert len(r['masks']['reference common']) <= len(ref)


def test_blend_flag():
    w, f, e = _spectrum([5010.0, 5020.0], [0.5, 0.5])
    t = lm.detect_lines(w, f)
    vald = pd.DataFrame({'wave': [5010.0, 5010.02, 5020.0, 5020.03], 'depth_vald': [0.5, 0.4, 0.5, 0.05], 'species': 'Fe 1'})
    t = lm.identify(t, vald, resolution=R_TEST)
    assert t['n_candidates'].tolist() == [2, 2]
    assert t['n_candidates_deep'].tolist() == [2, 1]          # the second neighbour is too weak to count
    assert t['n_blend'].tolist() == [1, 0]


def test_identify_prefers_close_deep_line():
    # 3801.814 A of the PoET table: a deep VALD line 5 mA from the minimum and a weaker one 48 mA away; the old score
    # took the far one because VALD depths (unbroadened) are deeper than the observed one
    t = pd.DataFrame({'wave_min': [3801.8138], 'depth': [0.31], 'wave_left_max': [3801.74], 'wave_right_max': [3801.89]})
    vald = pd.DataFrame({'wave': [3801.8086, 3801.862], 'depth_vald': [0.86, 0.716], 'species': ['Fe 1', 'Mg 1']})
    assert lm.identify(t, vald, score='window')['species'].iloc[0] == 'Mg 1'
    assert lm.identify(t, vald, resolution=R_TEST, vald_depth_scale=0.85)['species'].iloc[0] == 'Fe 1'
    # a close VALD line far too weak for the observed depth loses to a deep one 1 resolution element away
    vald = pd.DataFrame({'wave': [5000.002, 5000.0263], 'depth_vald': [0.02, 0.6], 'species': ['weak', 'deep']})
    t = pd.DataFrame({'wave_min': [5000.0], 'depth': [0.5], 'wave_left_max': [4999.9], 'wave_right_max': [5000.1]})
    assert lm.identify(t, vald, resolution=R_TEST, vald_depth_scale=1.0)['species'].iloc[0] == 'deep'
    with pytest.raises(ValueError):
        lm.identify(t, vald)


def test_template_fit_recovers_shift():
    c, d = _lines()
    rv = lm.RV_GRID
    ccfs = []
    for v in (-3.0, 0.0, 2.0, 5.0):
        w, f, _ = _spectrum(c * (1 + v / C), d)
        ccfs.append(lm.ccf(w, f, c, d))
    ref = lm.template_reference(np.median(ccfs, axis=0), rv)
    t = lm.template_fit(ref, np.array(ccfs))
    assert np.allclose(t['rv'] - t['rv'][1], [-3.0, 0.0, 2.0, 5.0], atol=0.2)


def test_same_wavelength_components_are_not_blends():
    w, f, e = _spectrum([5010.0], [0.5])
    t = lm.detect_lines(w, f)
    vald = pd.DataFrame({'wave': [5010.0, 5010.002, 5010.004], 'depth_vald': [0.5, 0.5, 0.5], 'species': 'Mn 1'})
    t = lm.identify(t, vald, resolution=R_TEST)
    assert t['n_candidates_deep'].tolist() == [3] and t['n_blend'].tolist() == [0]


def test_line_presence_window():
    centres, depths = _lines()
    ref = _spectrum(centres, depths)[:2]
    shift = 150.0
    spectra = [_spectrum(centres * (1 + shift / C), depths, noise=0.004, seed=s)[:2] for s in range(5)]
    missing = _spectrum(np.delete(centres, 3) * (1 + shift / C), np.delete(depths, 3), noise=0.004, seed=9)[:2]
    frac = lm.line_presence(spectra + [missing], centres, reference=ref, prominence=0.025, ref_prominence=0.05, shift=shift)
    assert np.all(frac[np.arange(len(centres)) != 3] == 1)
    assert np.isclose(frac[3], 5 / 6)


def test_presence_in_windows_and_flags():
    c, d = _lines(n=20)
    w, f, _ = _spectrum(c, d)
    t = lm.detect_lines(w, f, prominence=0.05)
    # second spectrum without the first three lines
    w2, f2, _ = _spectrum(c[3:], d[3:])
    n = lm.flags.presence_in_windows([(w, f), (w2, f2)], t['wave_left_max'], t['wave_right_max'], prominence=0.025)
    assert list(n[:3]) == [1, 1, 1] and np.all(n[3:] == 2)
    tab = pd.DataFrame({'a': [True, False, True], 'b': [False, False, True], 'obs_weight': [1.0, 1.0, 2.0]})
    assert list(lm.flags.first_flag(tab, ['b', 'a'])) == ['a', 'no flag', 'b']
    assert list(lm.flags.select(tab, ['a', 'b'])) == [False, True, False]
    s = lm.flags.summary(tab, ['a', 'b'])
    assert s.loc['a', 'lines'] == 2 and s.loc['a', 'only this flag'] == 1 and s.loc['no flag', 'lines'] == 1


def test_joint_telluric_fit_recovers_columns():
    # two molecules with gaussian optical-depth lines on a vacuum grid, two pressures, one temperature; molecule B is
    # stronger at the second pressure point; inject columns, recover them
    wv = np.arange(5990.0, 6110.0, 0.01)
    rng = np.random.default_rng(3)
    def tau(centres, width):
        return sum(np.exp(-0.5 * ((wv - x) / width) ** 2) for x in centres)
    ca, cb = np.sort(rng.uniform(6002, 6028, 12)), np.sort(rng.uniform(6072, 6098, 12))
    grid = {'wave_vac': wv, 'p': np.array([0.5, 1.0]), 'T': np.array([250.0]),
            'tau_A': np.array([[tau(ca, 0.01)], [tau(ca, 0.02)]]), 'tau_B': np.array([[tau(cb, 0.01)], [tau(cb, 0.02)]])}
    bands = {'A': ('A', (6000.0, 6030.0), None), 'B': ('B', (6070.0, 6100.0), None)}
    truth = {'scale_A': 0.3, 'scale_B': 0.8, 'ip_A': 1, 'iT_A': 0, 'ip_B': 0, 'iT_B': 0, 'R A': 1.9e5, 'R B': 1.9e5, 'dv': 150.0}
    wt, tt = lm.tellurics.telluric_model_joint(grid, truth, bands=bands, molecules=('A', 'B'))
    w = wt[(wt > 6000) & (wt < 6100)]
    obs = np.interp(w, wt, tt)
    best, allf = lm.fit_tellurics_joint(grid, w, obs, np.ones_like(w), bands=bands, molecules=('A', 'B'))
    assert best['ip_A'] == 1 and best['ip_B'] == 0
    assert abs(best['scale_A'] / 0.3 - 1) < 0.05 and abs(best['scale_B'] / 0.8 - 1) < 0.05
    assert abs(best['dv'] - 150.0) < 30.0


def test_curvature_edges_ignore_noise_bumps():
    # two lines 0.25 A apart with noise: the edges between them must be near the true maximum, not stopped by noise
    c, d = np.array([5020.0, 5020.25]), np.array([0.6, 0.5])
    w0, f0, _ = _spectrum(c, d)
    i_true = np.argmax(np.where((w0 > c[0]) & (w0 < c[1]), f0, -1))
    w, f, e = _spectrum(c, d, noise=2e-3, seed=4)
    first = lm.detect_lines(w, f, prominence=0.05)
    new = lm.detect_lines(w, f, prominence=0.05, edges='curvature', error=e)
    assert len(new) == 2
    r_new = new.sort_values('wave_min')['i_right'].values[0]
    l_new = new.sort_values('wave_min')['i_left'].values[1]
    px = 1.0 / np.median(np.diff(w)) * 0.02          # pixels in 0.02 A
    assert abs(r_new - i_true) < px and abs(l_new - i_true) < px
    # an isolated line on a flat noisy continuum: the window stops where the wing reaches the continuum
    w, f, e = _spectrum(np.array([5025.0]), np.array([0.5]), noise=2e-3, seed=5)
    t = lm.detect_lines(w, f, prominence=0.05, edges='curvature', error=e)
    width = t['wave_right_max'].values[0] - t['wave_left_max'].values[0]
    sigma = 5025.0 * 3000.0 / C
    assert 3 * sigma < width < 12 * sigma


def test_trace_gases_follow_o2_column():
    best = {'scale_O2': 200.0, 'p_O2': 0.5, 'T_O2': 250.0, 'ip_O2': 1, 'iT_O2': 0}
    out = lm.add_trace_gases(best, mixing={'CO2': 4.19e-4})
    assert out['scale_CO2'] == pytest.approx(200.0 * 4.19e-4 / 0.2095)
    assert (out['ip_CO2'], out['iT_CO2']) == (1, 0)
    assert 'scale_CO2' not in best
