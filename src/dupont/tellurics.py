"""Telluric model fitted to an observed spectrum, and selection of the lines free of telluric absorption.

The model: HITRAN optical depths of H2O and O2 (Voigt profiles on a small pressure x temperature grid), scaled by a
column factor, convolved to the instrument resolution and shifted in velocity, fitted to observed / synthetic (the
synthetic spectrum has no tellurics, so the ratio leaves the telluric lines). H2O is fitted jointly on three regions
and used everywhere; each O2 band is fitted on its own, with that H2O (fit_tellurics), or every band at once
(fit_tellurics_joint). fetch_hitran gets every isotopologue of every molecule with optical lines; add_trace_gases adds
the trace gases (CO2, CH4, N2O, CO, NO, H2) with their mixing ratios to the air column given by the fitted O2.
"""
import os

import numpy as np
import pandas as pd

from .spectra import C_LIGHT, vacuum_to_air


# ---------------------------------------------------------------------------------------------------------------------
# line selection
# ---------------------------------------------------------------------------------------------------------------------
def telluric_depth(wave_left, wave_right, tell_wave, tell_trans, dv_range=(-1000.0, 1000.0)):
    """Largest telluric depth (1 - transmission) inside the window of every line, the window widened by the velocity
    range dv_range [m/s] over which the tellurics move relative to the lines (Earth's motion; for the Sun, about
    +-1 km/s over the year, much less within a night). tell_wave, tell_trans: the model in the frame of the lines."""
    lo = np.asarray(wave_left) * (1.0 + dv_range[0] / C_LIGHT)
    hi = np.asarray(wave_right) * (1.0 + dv_range[1] / C_LIGHT)
    i0 = np.searchsorted(tell_wave, lo)
    i1 = np.maximum(np.searchsorted(tell_wave, hi), i0 + 1)
    i1 = np.minimum(i1, len(tell_wave))
    i0 = np.minimum(i0, i1 - 1)
    # running minimum of the transmission over every window
    return np.array([1.0 - np.min(tell_trans[a:b]) for a, b in zip(i0, i1)])


# ---------------------------------------------------------------------------------------------------------------------
# HITRAN optical depths
# ---------------------------------------------------------------------------------------------------------------------
def hitran_tau_grid(path_hitran, molecules=('H2O', 'O2'), pressures=(0.25, 0.5, 0.75, 1.0), temperatures=(250.0, 296.0),
                    wave=np.arange(3790.0, 7960.0 + 1e-6, 0.01), length=1e3, cache=None):
    """Optical depth of every molecule on the vacuum wavelengths wave [A] for every pressure [atm] and temperature [K]:
    HITRAN line lists already fetched in path_hitran (with hapi), Voigt profiles, a path of `length` [cm] of the pure
    gas. Returns {'wave_vac', 'p', 'T', 'tau_<molecule>' (n_p, n_T, n_wave)}; read from / written to cache (.npz)."""
    if cache and os.path.exists(cache):
        with np.load(cache) as z:
            if (np.array_equal(z['wave_vac'], wave) and np.array_equal(z['p'], pressures)
                    and np.array_equal(z['T'], temperatures) and all('tau_' + m in z for m in molecules)):
                return dict(z)
    from hapi import db_begin, absorptionCoefficient_Voigt
    db_begin(str(path_hitran))
    out = {'wave_vac': wave, 'p': np.asarray(pressures, float), 'T': np.asarray(temperatures, float)}
    nu_range = [1e8 / wave[-1] - 1.0, 1e8 / wave[0] + 1.0]
    for mol in molecules:
        tau = np.zeros((len(pressures), len(temperatures), len(wave)))
        for i, p in enumerate(pressures):
            for j, T in enumerate(temperatures):
                nu, coef = absorptionCoefficient_Voigt(SourceTables=mol, Environment={'p': p, 'T': T}, HITRAN_units=False,
                                                       WavenumberRange=nu_range, WavenumberStep=0.01)
                tau[i, j] = np.interp(wave, (1e8 / nu)[::-1], (coef * length)[::-1], left=0, right=0)
        out['tau_' + mol] = tau
    if cache:
        np.savez(cache, **out)
    return out


# HITRAN molecule numbers with lines in the optical (3790-7960 A; O3 and NO2 only have cross sections there)
HITRAN_MOLECULES = {'H2O': 1, 'CO2': 2, 'N2O': 4, 'CO': 5, 'CH4': 6, 'O2': 7, 'NO': 8, 'H2': 45}
# dry-air mole fractions of the trace gases (NOAA global means, 2025-2026; NO: upper value of the remote troposphere)
MIXING_RATIOS = {'CO2': 425e-6, 'CH4': 1.93e-6, 'N2O': 0.337e-6, 'CO': 0.1e-6, 'NO': 1e-9, 'H2': 0.53e-6}
O2_FRACTION = 0.2095


def fetch_hitran(path_hitran, molecules=tuple(HITRAN_MOLECULES), wave=(3790.0, 7960.0), suffix='_all'):
    """Fetch from hitran.org (hapi) the lines of every isotopologue of each molecule between the vacuum wavelengths
    wave [A], as tables '<molecule><suffix>' in path_hitran; tables already there are not fetched again. HITRAN line
    intensities include the natural abundance of each isotopologue, so a table is the natural mixture."""
    import hapi
    os.makedirs(path_hitran, exist_ok=True)
    hapi.db_begin(str(path_hitran))
    numin, numax = 1e8 / wave[1] - 1.0, 1e8 / wave[0] + 1.0
    for mol in molecules:
        name = mol + suffix
        if name in hapi.tableList():
            continue
        M = HITRAN_MOLECULES[mol]
        ids = [v[hapi.ISO_INDEX['id']] for (m, i), v in hapi.ISO.items() if m == M]
        hapi.fetch_by_ids(name, ids, numin, numax)
    return [mol + suffix for mol in molecules]


def add_trace_gases(best, mixing=MIXING_RATIOS, parent='O2'):
    """Columns of the trace gases for telluric_model_joint, from the fitted column of the parent molecule (O2, well
    mixed): scale_m = mixing_m / O2_FRACTION x scale_O2, at the pressure and temperature of O2 (the optical depths of
    hitran_tau_grid are for the same path of pure gas, so the column ratio is the scale ratio). Returns a new dict."""
    out = dict(best)
    for m, x in mixing.items():
        out['scale_' + m] = x / O2_FRACTION * best['scale_' + parent]
        for k in ('p_', 'T_', 'ip_', 'iT_'):
            out[k + m] = best[k + parent]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# model and fit
# ---------------------------------------------------------------------------------------------------------------------
# band: (molecule, region of the fit [A, air], region where the band's own fit is used in the final model)
BANDS = {'O2 gamma': ('O2', (6276.0, 6320.0), (6270.0, 6330.0)),
         'O2 B': ('O2', (6865.0, 6920.0), (6860.0, 6960.0)),
         'O2 A': ('O2', (7590.0, 7700.0), (7585.0, 7720.0)),
         'H2O 5900': ('H2O', (5880.0, 5990.0), None),
         'H2O 6500': ('H2O', (6470.0, 6560.0), None),
         'H2O 7250': ('H2O', (7160.0, 7340.0), None)}


def _convolve(w, t, resolution):
    """Transmission t on the uniform wavelengths w convolved with a gaussian of this resolution (at the centre of w)."""
    from scipy.ndimage import gaussian_filter1d
    sigma = np.mean(w) / resolution / (2 * np.sqrt(2 * np.log(2))) / np.median(np.diff(w))
    return gaussian_filter1d(t, sigma, mode='nearest')


def _convolve_chunks(w, t, resolution, chunk=250.0):
    out = np.empty_like(t)
    for lo in np.arange(w[0], w[-1], chunk):
        k = (w >= lo) & (w < lo + chunk)
        pad = (w >= lo - 5.0) & (w < lo + chunk + 5.0)
        out[k] = _convolve(w[pad], t[pad], resolution)[k[pad]]
    return out


def band_model(grid, mol, ip, iT, scale, resolution, dv, region, other=None):
    """Transmission of one molecule (pressure index ip, temperature index iT, column factor scale) in region [A, air]
    +- 3 A, times `other` (another transmission on the same grid, before convolution), convolved, in air, shifted by
    dv [m/s]. Returns (wave_air, transmission)."""
    w = grid['wave_vac']
    k = (w > region[0] - 1.5) & (w < region[1] + 5.5)          # vacuum - air ~ 1.8-2.2 A
    t = np.exp(-scale * grid['tau_' + mol][ip, iT, k])
    if other is not None:
        t = t * other[k]
    return vacuum_to_air(w[k]) * (1 + dv / C_LIGHT), _convolve(w[k], t, resolution)


def _local_continuum(x, y, m):
    """Straight line c(x) with y ~ m c: the local error of the continuum under a band (fitted where m > 0.3)."""
    ok = m > 0.3
    if ok.sum() < 10:
        return np.ones_like(x)
    return np.polyval(np.polyfit(x[ok] - x.mean(), y[ok] / m[ok], 1), x - x.mean())


def fit_band(grid, wave, ratio, mol, region, select, other=None, x0=(None, 1.9e5, 0.0)):
    """Fit of one band to ratio = observed / synthetic on the pixels `select` inside region [A, air]: for every
    pressure and temperature of the grid, robust least squares (soft L1) on log10(column factor), resolution and
    velocity shift [m/s], with a local straight continuum. Returns the best fit (dict) and all the fits (DataFrame)."""
    from scipy.optimize import least_squares
    sel = (wave > region[0]) & (wave < region[1]) & select
    x, y = wave[sel], ratio[sel]
    fits = []
    for ip, p in enumerate(grid['p']):
        for iT, T in enumerate(grid['T']):
            def resid(q):
                wt, ft = band_model(grid, mol, ip, iT, 10 ** q[0], q[1] * 1e5, q[2] * 1e3, region, other)
                m = np.interp(x, wt, ft)
                return y - m * _local_continuum(x, y, m)
            s0 = x0[0]
            if s0 is None:
                trial = np.logspace(-2.5, 3.5, 25)
                s0 = trial[np.argmin([np.sum(np.abs(resid([np.log10(s), x0[1] / 1e5, x0[2] / 1e3]))) for s in trial])]
            r = least_squares(resid, [np.log10(s0), x0[1] / 1e5, x0[2] / 1e3], loss='soft_l1', f_scale=0.01,
                              bounds=([-4, 0.6, -2.0], [5, 4.0, 2.0]))
            res = resid(r.x)
            wt, ft = band_model(grid, mol, ip, iT, 10 ** r.x[0], r.x[1] * 1e5, r.x[2] * 1e3, region, other)
            absorbed = np.interp(x, wt, ft) < 0.99
            fits.append({'p': p, 'T': T, 'ip': ip, 'iT': iT, 'scale': 10 ** r.x[0], 'resolution': r.x[1] * 1e5,
                         'dv': r.x[2] * 1e3, 'cost': r.cost,
                         'rms': np.sqrt(np.mean(res[absorbed] ** 2)) if absorbed.any() else np.nan,
                         'rms_all': np.sqrt(np.mean(res ** 2)), 'pixels': int(sel.sum()), 'absorbed': int(absorbed.sum())})
    fits = pd.DataFrame(fits)
    return fits.loc[fits['cost'].idxmin()].to_dict(), fits


def fit_tellurics(grid, wave, observed, synthetic, bands=BANDS, continuum=0.95, x0=(None, 1.9e5, 0.0)):
    """Telluric fit of observed / synthetic (same wavelengths, both flattened), on the pixels where the synthetic
    spectrum is above `continuum` (outside the cores of the solar lines). Every H2O region is fitted, then H2O jointly
    (same pressure and temperature; median column, resolution and shift), then every O2 band with that H2O.
    Returns a table of the fits (one row per band + 'H2O (joint)') and {band: all the fits of the grid}."""
    ratio = observed / synthetic
    select = synthetic > continuum
    rows, allfits = {}, {}
    h2o = [b for b, (m, _, _) in bands.items() if m == 'H2O']
    for band in h2o:
        best, allfits[band] = fit_band(grid, wave, ratio, 'H2O', bands[band][1], select, x0=x0)
        rows[band] = dict(best, molecule='H2O')
    joint = []
    for ip in range(len(grid['p'])):
        for iT in range(len(grid['T'])):
            c = sum(allfits[b].loc[(allfits[b]['ip'] == ip) & (allfits[b]['iT'] == iT), 'cost'].values[0] for b in h2o)
            joint.append((c, ip, iT))
    _, ip, iT = min(joint)
    sub = pd.concat([allfits[b][(allfits[b]['ip'] == ip) & (allfits[b]['iT'] == iT)] for b in h2o])
    rows['H2O (joint)'] = {'molecule': 'H2O', 'p': grid['p'][ip], 'T': grid['T'][iT], 'ip': ip, 'iT': iT,
                           'scale': float(np.exp(np.median(np.log(sub['scale'])))),
                           'resolution': float(np.median(sub['resolution'])), 'dv': float(np.median(sub['dv']))}
    h = rows['H2O (joint)']
    other = np.exp(-h['scale'] * grid['tau_H2O'][ip, iT])
    for band, (mol, region, _) in bands.items():
        if mol == 'H2O':
            continue
        best, allfits[band] = fit_band(grid, wave, ratio, mol, region, select, other=other, x0=x0)
        rows[band] = dict(best, molecule=mol)
    return pd.DataFrame(rows).T, allfits


def telluric_model(grid, fits, bands=BANDS):
    """Final transmission (wave_air, transmission) on the grid of hitran_tau_grid: the joint H2O fit everywhere, times
    every O2 band with its own fit inside its region."""
    w = grid['wave_vac']
    h = fits.loc['H2O (joint)']
    t_h2o = np.exp(-h['scale'] * grid['tau_H2O'][int(h['ip']), int(h['iT'])])
    w_air = vacuum_to_air(w)
    out = np.interp(w_air, w_air * (1 + h['dv'] / C_LIGHT), _convolve_chunks(w, t_h2o, h['resolution']))
    for band, (mol, _, use) in bands.items():
        if mol == 'H2O' or band not in fits.index:
            continue
        f = fits.loc[band]
        wb, tb = band_model(grid, mol, int(f['ip']), int(f['iT']), f['scale'], f['resolution'], f['dv'], use, other=t_h2o)
        k = (w_air >= use[0]) & (w_air <= use[1])
        out[k] = np.interp(w_air[k], wb, tb)
    return w_air, out


def fit_quality(wave, observed, synthetic, models, bands=BANDS, continuum=0.95):
    """rms of observed / synthetic - model x local continuum in every band (pixels with synthetic > continuum), on the
    pixels where the model absorbs more than 1 % and on all of them, for every model {name: (wave, transmission)}."""
    rows = {}
    for band, (mol, region, _) in bands.items():
        sel = (wave > region[0]) & (wave < region[1]) & (synthetic > continuum)
        y = observed[sel] / synthetic[sel]
        row = {}
        for name, (wt, ft) in models.items():
            m = np.interp(wave[sel], wt, ft)
            a = m < 0.99
            d = y - m * _local_continuum(wave[sel], y, m)
            row['%s: rms absorbed' % name] = np.sqrt(np.mean(d[a] ** 2)) if a.any() else np.nan
            row['%s: rms all' % name] = np.sqrt(np.mean(d ** 2))
        rows[band] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------------------------------------------------
# joint fit: every molecule in every band at once
# ---------------------------------------------------------------------------------------------------------------------
MOLECULES = ('H2O', 'O2')


def joint_model(grid, idx, scales, resolution, dv, region):
    """Transmission of all the molecules together in region [A, air] +- 3 A: exp(-sum_m scale_m tau_m(p_m, T_m)) with
    (p_m, T_m) = grid indices idx[m], convolved with a gaussian of this resolution, in air, shifted by dv [m/s].
    Returns (wave_air, transmission)."""
    w = grid['wave_vac']
    k = (w > region[0] - 1.5) & (w < region[1] + 5.5)          # vacuum - air ~ 1.8-2.2 A
    tau = sum(scales[m] * grid['tau_' + m][idx[m][0], idx[m][1], k] for m in scales)
    return vacuum_to_air(w[k]) * (1 + dv / C_LIGHT), _convolve(w[k], np.exp(-tau), resolution)


def fit_tellurics_joint(grid, wave, observed, synthetic, bands=BANDS, molecules=MOLECULES, continuum=0.95,
                        x0=(1.9e5, 0.0), f_scale=0.01, resolution_per_band=True):
    """Simultaneous telluric fit of observed / synthetic in all the bands: every molecule is in every band, with one
    column factor per molecule (the same atmosphere everywhere), one velocity shift (the same wavelength frame), the
    resolution free in every band (resolution_per_band; the ESPRESSO resolution changes along the spectrum) or common
    to all, and one straight local continuum per band (solved linearly at every step, not a free parameter of the
    optimiser). Pixels: inside the bands, where the synthetic spectrum is above `continuum` (outside the cores of the
    solar lines).

    Optimiser: for every combination of the pressure x temperature grid points of the molecules (the HITRAN optical
    depths are only known on that grid), scipy least_squares (trust region reflective) with a soft-L1 loss of scale
    f_scale (residuals beyond ~f_scale count linearly, so that solar-line residuals do not drive the fit) on
    [log10 column of every molecule, resolution(s) / 1e5, dv / 1e3]; bounds: column 1e-4 to 1e5, resolution 60 000 to
    400 000, dv +-2 km/s. Starting columns: a scan of 25 values of each molecule on its own bands. The best grid
    combination is the one with the smallest cost (sum of the soft-L1 residuals of all the bands together).
    Returns (best: dict, all: DataFrame, one row per grid combination, sorted by cost)."""
    import itertools
    from scipy.optimize import least_squares
    ratio = observed / synthetic
    sel_all = synthetic > continuum
    names = list(bands)
    data = {}
    for band, (mol, region, _) in bands.items():
        s = (wave > region[0]) & (wave < region[1]) & sel_all
        data[band] = (wave[s], ratio[s], region, mol)
    combos = list(itertools.product(range(len(grid['p'])), range(len(grid['T']))))
    nm = len(molecules)
    nr = len(names) if resolution_per_band else 1

    def unpack(q):
        scales = {m: 10 ** q[j] for j, m in enumerate(molecules)}
        res = {b: q[nm + (k if resolution_per_band else 0)] * 1e5 for k, b in enumerate(names)}
        return scales, res, q[nm + nr] * 1e3

    def resid(q, idx, which=None):
        scales, res, dv = unpack(q)
        out = []
        for band, (x, y, region, _) in data.items():
            if which is not None and band not in which:
                continue
            wt, ft = joint_model(grid, idx, scales, res[band], dv, region)
            m = np.interp(x, wt, ft)
            out.append(y - m * _local_continuum(x, y, m))
        return np.concatenate(out)

    rows = []
    for combo in itertools.product(combos, repeat=nm):
        idx = dict(zip(molecules, combo))
        # starting columns: each molecule alone on its own bands (the others at 0), 25 values
        q0 = [-10.0] * nm + [x0[0] / 1e5] * nr + [x0[1] / 1e3]
        for j, m in enumerate(molecules):
            own = [b for b, (mol, _, _) in bands.items() if mol == m]
            trial = np.linspace(-2.5, 3.5, 25)
            cost = []
            for s in trial:
                q = list(q0)
                q[j] = s
                cost.append(np.sum(np.abs(resid(q, idx, own))))
            q0[j] = trial[int(np.argmin(cost))]
        r = least_squares(resid, q0, args=(idx,), loss='soft_l1', f_scale=f_scale,
                          bounds=([-4.0] * nm + [0.6] * nr + [-2.0], [5.0] * nm + [4.0] * nr + [2.0]))
        scales, res, dv = unpack(r.x)
        row = {'cost': r.cost, 'nfev': r.nfev, 'dv': dv}
        for j, m in enumerate(molecules):
            row.update({'p_' + m: grid['p'][idx[m][0]], 'T_' + m: grid['T'][idx[m][1]], 'ip_' + m: idx[m][0],
                        'iT_' + m: idx[m][1], 'scale_' + m: scales[m]})
        for b in names:
            row['R ' + b] = res[b]
        rows.append(row)
    allfits = pd.DataFrame(rows).sort_values('cost').reset_index(drop=True)
    return allfits.iloc[0].to_dict(), allfits


def _resolution_at(best, bands, wave):
    """Resolution of the joint fit at the wavelengths `wave`: interpolated between the band centres (constant beyond)."""
    keys = [b for b in bands if 'R ' + b in best]
    c = np.array([0.5 * sum(bands[b][1]) for b in keys])
    r = np.array([best['R ' + b] for b in keys])
    o = np.argsort(c)
    return np.interp(wave, c[o], r[o])


def telluric_model_joint(grid, best, bands=BANDS, molecules=MOLECULES, chunk=50.0):
    """Transmission (wave_air, transmission) of the joint fit on the whole grid of hitran_tau_grid: the fitted
    columns everywhere, convolved in chunks of `chunk` [A] with the resolution of the band (interpolated between the
    band centres, constant beyond them), shifted by the fitted dv."""
    from scipy.ndimage import gaussian_filter1d
    w = grid['wave_vac']
    tau = sum(best['scale_' + m] * grid['tau_' + m][int(best['ip_' + m]), int(best['iT_' + m])] for m in molecules)
    t0 = np.exp(-tau)
    w_air = vacuum_to_air(w)
    t = np.empty_like(t0)
    step = np.median(np.diff(w))
    for lo in np.arange(w[0], w[-1], chunk):
        k = (w >= lo) & (w < lo + chunk)
        pad = (w >= lo - 5.0) & (w < lo + chunk + 5.0)
        R = float(_resolution_at(best, bands, vacuum_to_air(lo + 0.5 * chunk))) if 'R ' + next(iter(bands)) in best else best['resolution']
        sigma = (lo + 0.5 * chunk) / R / (2 * np.sqrt(2 * np.log(2))) / step
        t[k] = gaussian_filter1d(t0[pad], sigma, mode='nearest')[k[pad]]
    return w_air, np.interp(w_air, w_air * (1 + best['dv'] / C_LIGHT), t)
