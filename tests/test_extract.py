"""Tests of extract_mask (one or more spectra), the comparison of masks and the spectrum tools of prepare."""
import numpy as np
import pandas as pd
import pytest

import dupont as lm

C = lm.C_LIGHT
R = 190000.0


def _spectrum(centres, depths, sigma_v=3000.0, wave=None):
    if wave is None:
        wave = np.exp(np.arange(np.log(5000.0), np.log(5050.0), 1.0 / 600000))
    f = np.ones_like(wave)
    for c, d in zip(centres, depths):
        f -= d * np.exp(-0.5 * ((wave - c) / (c * sigma_v / C)) ** 2)
    return wave, f


def _lines(n=40, seed=1):
    rng = np.random.default_rng(seed)
    c = np.sort(rng.uniform(5002, 5048, n))
    c = c[np.concatenate([[True], np.diff(c) > 0.4])]
    return c, rng.uniform(0.1, 0.8, len(c))


def _vald(c, d):
    return pd.DataFrame({'wave': c, 'depth_vald': d, 'species': 'Fe 1'})


def test_one_spectrum():
    c, d = _lines()
    w, f = _spectrum(c, d)
    r = lm.extract_mask((w, f))
    assert r.names == ['spectrum']
    assert len(r.masks['all']) == len(c)
    assert np.max(np.abs(C * (r.masks['all']['wave'].values - c) / c)) < 5.0
    assert np.isclose(r.masks['all']['weight'].sum(), 1.0)
    # weights: Bouchy eq. 8 of the spectrum over each line window
    t = lm.rv_weights(w, f, None, lm.detect_lines(w, f))
    assert np.allclose(np.sort(r.lines['weight']), np.sort(t['weight']))


def test_two_spectra_shift_and_grid():
    c, d = _lines()
    w, f = _spectrum(c, d)
    w2 = np.linspace(5000.0, 5050.0, 120000)                  # another grid
    _, f2 = _spectrum(c * (1 + 400.0 / C), d * 0.9, wave=w2)  # shifted, shallower
    r = lm.extract_mask({'obs': (w, f), 'syn': (w2, f2)})
    assert abs(r.shifts['syn'] - 400.0) < 10.0
    assert len(r.lines) == len(c)
    assert np.max(np.abs(r.lines['dv_syn'])) < 30.0
    assert {'obs_weight', 'syn_weight', 'obs_wave_min', 'syn_wave_min'} <= set(r.lines.columns)
    # the mask is at the primary wavelengths; primary='syn' puts it at the shifted ones
    r2 = lm.extract_mask({'obs': (w, f), 'syn': (w2, f2)}, primary='syn')
    assert np.median(C * (r2.masks['all']['wave'].values - c) / c) == pytest.approx(400.0, abs=10.0)


def test_line_missing_in_one_spectrum_is_dropped():
    c, d = _lines()
    w, f = _spectrum(c, d)
    _, f2 = _spectrum(c[1:], d[1:], wave=w)
    _, f3 = _spectrum(c[:-1], d[:-1], wave=w)
    r = lm.extract_mask({'a': (w, f), 'b': (w, f2), 'c': (w, f3)})
    assert len(r.lines) == len(c) - 2
    assert r.counts['lines in every spectrum'] == len(c) - 2


def test_same_as_build_mask():
    c, d = _lines()
    w, f = _spectrum(c, d)
    _, f2 = _spectrum(c * (1 + 150.0 / C), d * 0.95, sigma_v=2600.0, wave=w)
    e = np.full_like(w, 1e-3)
    old = lm.build_mask((w, f, e), (w, f2), _vald(c, d), lm.MaskConfig(names=('obs', 'syn'), resolution=R))
    new = lm.extract_mask({'obs': (w, f, e), 'syn': (w, f2)}, lm.ExtractConfig(resolution=R), vald=_vald(c, d))
    a, b = old['masks']['obs all'], new.masks['all']
    assert len(a) == len(b)
    assert np.allclose(a['wave'], b['wave']) and np.allclose(a['weight'], b['weight'])


def test_flags_and_custom_masks():
    c, d = _lines()
    w, f = _spectrum(c, d)
    tw = np.linspace(4990, 5060, 20000)
    tt = np.ones_like(tw)
    tt[np.abs(tw - c[3]) < 0.02] = 0.9                             # a telluric line on line 3
    r = lm.extract_mask((w, f), lm.ExtractConfig(min_depth=0.3, resolution=R), vald=_vald(c[1:], d[1:]), telluric=(tw, tt))
    t = r.lines
    assert t['telluric'].sum() == 1 and t.loc[t['telluric'], 'wave'].iloc[0] == pytest.approx(c[3], abs=0.01)
    assert t['shallow'].sum() == np.sum(d < 0.3)
    assert t['no_vald'].sum() == 1
    assert len(r.masks['all']) == len(c) - 1 - np.sum((d < 0.3) & (np.arange(len(c)) != 3))
    assert len(r.mask(exclude=())) == len(c)
    assert len(r.mask(exclude=('no_vald',))) == len(c) - 1


def test_compare_masks():
    c, d = _lines()
    m = pd.DataFrame({'wave': c, 'weight': d ** 2 / np.sum(d ** 2)})
    ref = pd.DataFrame({'wave': c[5:] * (1 + 500.0 / C), 'weight': d[5:] ** 2})
    out = lm.compare_masks(m, ref, max_dv=1000.0, shift=None)
    assert out['shift'] == pytest.approx(500.0, abs=1e-6)
    assert out['counts']['matched lines'] == len(c) - 5
    assert out['counts']['weight r'] == pytest.approx(1.0)
    assert np.allclose(out['matched']['dv'], 0.0, atol=1e-6)
    assert out['counts']['reference weight matched [%]'] == pytest.approx(100.0)


def test_read_mask_formats(tmp_path):
    c, d = _lines()
    pd.DataFrame({'wave': c, 'cont': d}).to_csv(tmp_path / 'm.csv', index=False)
    m = lm.read_mask(tmp_path / 'm.csv', power=2)
    assert np.allclose(m['weight'], d ** 2 / np.sum(d ** 2))
    np.savetxt(tmp_path / 'm.mas', np.column_stack([c - 0.01, c + 0.01, d]))
    m2 = lm.read_mask(tmp_path / 'm.mas')
    assert np.allclose(m2['wave'], c) and np.allclose(m2['weight'], d / d.sum())


def test_compare_ccfs_rv():
    c, d = _lines()
    w, f = _spectrum(c * (1 + 1000.0 / C), d)
    m = pd.DataFrame({'wave': c, 'weight': d})
    t, curves = lm.compare_ccfs({'s': (w, f)}, {'m': m})
    assert t.loc[('s', 'm'), 'rv'] == pytest.approx(1000.0, abs=20.0)
    assert t.loc[('s', 'm'), 'lines used'] == len(c)


def test_resample_conserves_flux():
    c, d = _lines()
    w, f = _spectrum(c, d)
    nw = lm.velocity_grid(5005.0, 5045.0, 2000.0)          # 4 input pixels per pixel
    g = lm.resample(w, f, nw)
    k = (w > nw[0]) & (w < nw[-1])
    assert np.trapezoid(1 - g, nw) == pytest.approx(np.trapezoid(1 - f[k], w[k]), rel=2e-3)
    g2, e2 = lm.resample(w, f, nw, error=np.full_like(w, 0.01))
    assert np.all(e2 <= 0.0051) and np.allclose(g, g2, equal_nan=True)


def test_rotational_broadening():
    w, f = _spectrum([5025.0], [0.6], sigma_v=1500.0)
    g = lm.rotational_broadening(w, f, 10000.0)
    assert np.trapezoid(1 - g, w) == pytest.approx(np.trapezoid(1 - f, w), rel=1e-3)   # equivalent width
    assert g.min() > f.min() + 0.2                                                       # shallower, wider
    assert w[np.argmin(g)] == pytest.approx(5025.0, abs=0.005)


def test_add_noise():
    f = np.ones(200000)
    n, s = lm.add_noise(f, 200.0, seed=3)
    assert np.std(n - f) == pytest.approx(1 / 200.0, rel=0.02) and np.allclose(s, 1 / 200.0)


def test_disc_integrate_sunsim():
    pytest.importorskip('sunsim')
    w, f = _spectrum([5025.0], [0.6], sigma_v=1500.0)
    k = (w > 5015) & (w < 5035)
    spec = {mu: {'wav': w[k], 'intensity': (0.4 + 0.6 * mu) * f[k]} for mu in (0.2, 0.5, 1.0)}
    wd, fd = lm.disc_integrate(spec, n_rings=20)
    fd = fd / np.median(fd[:200])
    a, q = 1 - fd, np.abs(wd - 5025.0) < 0.2
    assert fd.min() > f[k].min() + 0.05                                                  # broadened by the rotation
    assert C * (np.sum(a[q] * wd[q]) / np.sum(a[q]) - 5025.0) / 5025.0 == pytest.approx(0.0, abs=5.0)   # centroid
    assert np.trapezoid(a, wd) == pytest.approx(np.trapezoid(1 - f[k], w[k]), rel=1e-3)                # EW


def test_write_mask_mas(tmp_path):
    c, d = _lines()
    m = pd.DataFrame({'wave': c, 'weight': d / d.sum()})
    lm.write_mask(m, tmp_path / 'm.mas')
    m2 = lm.read_mask(tmp_path / 'm.mas')
    assert np.allclose(m2['wave'], c, atol=1e-6) and np.allclose(m2['weight'], m['weight'], rtol=1e-5)


def test_shipped_masks():
    esp = lm.read_mask(lm.mask_path('ESPRESSO_SUN.mas'))
    csv = lm.load_reference_mask(lm.mask_path('ESPRESSO_SUN.csv'), weight='contrast2')
    assert len(esp) == len(csv) and np.allclose(esp['wave'], csv['wave']) and np.allclose(esp['weight'], csv['weight'], rtol=1e-5)
    poet = lm.read_mask(lm.mask_path('PoET_noflag.mas'))
    assert len(poet) > 3000 and np.isclose(poet['weight'].sum(), 1.0)
    with pytest.raises(ValueError):
        lm.mask_path('HARPS.mas')
