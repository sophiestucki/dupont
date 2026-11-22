"""Diagnostic figures of a mask (matplotlib). Every function draws one figure and returns it."""
import numpy as np
import matplotlib.pyplot as plt

from .spectra import C_LIGHT

COLORS = {'obs': '#cf5d3a', 'syn': '#45a5aa', 'ref': '#102e37', 'tell': '#e34948', 'grey': '0.6'}
SERIES = ['#102e37', '#cf5d3a', '#45a5aa', '#edae49', '#7b4fa0', '#66a182', '#d1495b', '#8c8c8c']


def _style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    try:
        ax.ticklabel_format(useOffset=False, style='plain', axis='x')
    except AttributeError:          # log axes
        pass
    ax.tick_params(direction='in', top=True, right=True)


def _dv(a, b):
    return C_LIGHT * (np.asarray(a) - np.asarray(b)) / np.asarray(b)


def spectra_zoom(spectra, lines=None, window=(5000.0, 5010.0), vald=None, names=None):
    """Spectra {label: (wave, flux)} in a wavelength window, with the minima of the mask lines (lines table of
    build_mask) and the VALD lines (depth bars)."""
    fig, ax = plt.subplots(figsize=(15, 4))
    for k, (label, (w, f)) in enumerate(spectra.items()):
        s = (w > window[0]) & (w < window[1])
        ax.plot(w[s], f[s], lw=1, color=SERIES[k], label=label)
    if vald is not None:
        v = vald[(vald['wave'] > window[0]) & (vald['wave'] < window[1])]
        ax.vlines(v['wave'], 1 - v['depth_vald'], 1, color='g', alpha=0.25, lw=0.8, label='VALD')
    if lines is not None:
        l = lines[(lines['obs_wave_min'] > window[0]) & (lines['obs_wave_min'] < window[1])]
        for sel, mk, lab in ((~l['blend'] & l['selected'], 'v', 'mask line'), (l['blend'] & l['selected'], 'v', 'mask line, blend'),
                             (~l['selected'], 'x', 'common line, not selected (telluric)')):
            ax.plot(l.loc[sel, 'obs_wave_min'], l.loc[sel, 'obs_flux_min'] - 0.03, mk, ms=5,
                    color={'mask line': 'k', 'mask line, blend': '#7b4fa0'}.get(lab, COLORS['tell']), label=lab)
    ax.set_xlim(*window)
    ax.set_xlabel(r'Wavelength [$\AA$]'); ax.set_ylabel('Normalised flux')
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _style(ax)
    return fig


def minimum_fit(lines):
    """Shift of the minimum from the core fit relative to the minimum pixel [m/s], for both spectra, and PoET minus
    MURaM minima."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    bins = np.linspace(-600, 600, 97)
    for p, c, lab in (('obs', COLORS['obs'], 'observed'), ('syn', COLORS['syn'], 'synthetic')):
        axes[0].hist(_dv(lines[p + '_wave_min'], lines[p + '_wave_min_pixel']), bins=bins, histtype='step', color=c, lw=1.5,
                     label=lab)
    axes[0].set_xlabel('core fit - minimum pixel [m/s]'); axes[0].set_ylabel('lines'); axes[0].legend(frameon=False)
    axes[1].hist(lines['dv'], bins=np.linspace(-2000, 2000, 81), color=COLORS['ref'])
    axes[1].set_xlabel('synthetic - observed minimum (global shift removed) [m/s]')
    for ax in axes:
        _style(ax)
    return fig


def vald_identification(lines):
    """Observed line against its VALD line: wavelength offset [m/s], depth against VALD depth, and the number of VALD
    candidates in the window (all, and deeper than the blend threshold)."""
    l = lines[lines['has_vald']]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    axes[0].hist(_dv(l['obs_wave_min'], l['vald_wave']), bins=np.linspace(-3000, 3000, 121), color=COLORS['ref'])
    axes[0].set_xlabel('observed minimum - VALD [m/s]')
    axes[1].plot(l['vald_depth'], l['obs_depth'], '.', ms=2, alpha=0.4, color=COLORS['obs'])
    axes[1].plot([0, 1], [0, 1], 'k--', lw=0.8)
    axes[1].set_xlabel('VALD depth'); axes[1].set_ylabel('observed depth')
    n = np.arange(0, 11)
    axes[2].bar(n - 0.2, [np.sum(lines['n_candidates'] == k) for k in n], 0.4, color=COLORS['grey'], label='all VALD lines')
    axes[2].bar(n + 0.2, [np.sum(lines['n_blend'] == k) for k in n], 0.4, color=COLORS['ref'],
                label='deep neighbours (blend if > 0)')
    axes[2].set_xlabel('VALD lines in the window'); axes[2].set_ylabel('lines'); axes[2].legend(frameon=False)
    for ax in axes:
        _style(ax)
    return fig


def blend_examples(lines, spectra, vald, n_show=8, half_window=0.25, seed=1):
    """Random blends: the observed and synthetic spectra around the line, the VALD lines of the window."""
    b = lines[lines['blend'] & lines['selected']]
    b = b.sample(min(n_show, len(b)), random_state=seed).sort_values('obs_wave_min')
    nc = 4
    fig, axes = plt.subplots(int(np.ceil(len(b) / nc)), nc, figsize=(4.2 * nc, 3.4 * np.ceil(len(b) / nc)), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), b.iterrows()):
        c = r['obs_wave_min']
        for k, (label, (w, f)) in enumerate(spectra.items()):
            s = np.abs(w - c) < half_window
            ax.plot(w[s], f[s], lw=1, color=SERIES[k + 1], label=label)
        v = vald[(vald['wave'] > r['obs_wave_left_max']) & (vald['wave'] < r['obs_wave_right_max'])]
        ax.vlines(v['wave'], 1 - v['depth_vald'], 1, color='g', alpha=0.6, lw=1)
        for _, vv in v.iterrows():
            ax.text(vv['wave'], 1.01, str(vv.get('species', '')), rotation=90, fontsize=7, ha='center', va='bottom')
        ax.axvspan(r['obs_wave_left_max'], r['obs_wave_right_max'], color='0.9', zorder=0)
        ax.axvline(c, color='k', lw=0.8, ls=':')
        ax.set_title('%.3f $\\AA$ %s, %d deep neighbour(s)' % (c, r['species'], r['n_blend']), fontsize=10)
        ax.set_ylim(None, 1.12)
        _style(ax)
    for ax in axes.ravel()[len(b):]:
        ax.axis('off')
    axes[0, 0].legend(frameon=False, fontsize=7)
    return fig


def windows_and_weights(lines, names=('observed', 'synthetic')):
    """Width of the windows of the weights (shared, and each line's own) and the weights of the two spectra."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    own_o = lines['obs_wave_right_max_own'] - lines['obs_wave_left_max_own']
    own_s = lines['syn_wave_right_max_own'] - lines['syn_wave_left_max_own']
    shared = lines['obs_wave_right_max'] - lines['obs_wave_left_max']
    bins = np.linspace(0, 1.0, 101)
    axes[0].hist(own_o, bins, histtype='step', lw=1.5, color=COLORS['obs'], label='%s, own window' % names[0])
    axes[0].hist(own_s, bins, histtype='step', lw=1.5, color=COLORS['syn'], label='%s, own window' % names[1])
    axes[0].hist(shared, bins, histtype='step', lw=1.5, color='k', label='shared window')
    axes[0].set_xlabel(r'window width [$\AA$]'); axes[0].legend(frameon=False)
    sel = lines['selected']
    ax = axes[1]
    ax.loglog(lines.loc[sel, 'obs_weight'], lines.loc[sel, 'syn_weight'], '.', ms=2, alpha=0.4, color=COLORS['ref'])
    lim = np.array([lines['obs_weight'][lines['obs_weight'] > 0].min(), lines['obs_weight'].max()])
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_xlabel('%s weight' % names[0]); ax.set_ylabel('%s weight' % names[1])
    for a in axes:
        _style(a)
    return fig


def small_synthetic_weight(lines, spectra, vald, n_show=8, half_window=0.3):
    """The selected lines whose synthetic weight is smallest relative to the observed one."""
    l = lines[lines['selected'] & (lines['obs_weight'] > 0)].copy()
    l['ratio'] = l['syn_weight'] / l['obs_weight']
    l = l.nsmallest(n_show, 'ratio').sort_values('obs_wave_min')
    nc = 4
    fig, axes = plt.subplots(int(np.ceil(len(l) / nc)), nc, figsize=(4.2 * nc, 3.4 * np.ceil(len(l) / nc)), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), l.iterrows()):
        c = r['obs_wave_min']
        for k, (label, (w, f)) in enumerate(spectra.items()):
            s = np.abs(w - c) < half_window
            ax.plot(w[s], f[s], lw=1, color=SERIES[k + 1], label=label)
        v = vald[np.abs(vald['wave'] - c) < half_window]
        ax.vlines(v['wave'], 1 - v['depth_vald'], 1, color='g', alpha=0.4, lw=0.8)
        ax.axvspan(r['obs_wave_left_max'], r['obs_wave_right_max'], color='0.9', zorder=0)
        ax.set_title('%.3f $\\AA$ %s: ratio %.2f' % (c, r['species'], r['ratio']), fontsize=10)
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7)
    return fig


def reference_comparison(lines, ref_name='ESPRESSO'):
    """Lines in the reference mask: weights (normalised over these lines) and wavelengths against the reference."""
    l = lines[lines['in_reference']]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    wo = l['obs_weight'] / l['obs_weight'].sum() * len(l)
    wr = l['ref_weight'] / l['ref_weight'].sum() * len(l)
    axes[0].loglog(wr, wo, '.', ms=2, alpha=0.4, color=COLORS['ref'])
    lim = [min(wr.min(), wo[wo > 0].min()), max(wr.max(), wo.max())]
    axes[0].plot(lim, lim, 'k--', lw=0.8)
    axes[0].set_xlabel('%s weight' % ref_name); axes[0].set_ylabel('observed weight')
    axes[1].plot(l['obs_wave_min'], wo - wr, '.', ms=2, alpha=0.4, color=COLORS['obs'])
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_xlabel(r'Wavelength [$\AA$]'); axes[1].set_ylabel('observed - %s weight' % ref_name)
    for p, c, lab in (('obs', COLORS['obs'], 'observed'), ('syn', COLORS['syn'], 'synthetic')):
        axes[2].hist(_dv(l[p + '_wave_min'], l['ref_wave']), bins=np.linspace(-3000, 3000, 121), histtype='step', lw=1.5,
                     color=c, label=lab)
    axes[2].set_xlabel('minimum - %s wavelength [m/s]' % ref_name); axes[2].legend(frameon=False)
    for ax in axes:
        _style(ax)
    return fig


def selection_along_wavelength(lines, reference, step=100.0):
    """Number of lines per wavelength bin: common lines, selected, telluric, blends, and the reference mask (all and
    matched)."""
    edges = np.arange(np.floor(lines['obs_wave_min'].min() / step) * step, lines['obs_wave_min'].max() + step, step)
    h = lambda x: np.histogram(x, edges)[0]
    mid = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(15, 4))
    for x, lab, c in ((lines['obs_wave_min'], 'common lines', COLORS['grey']),
                      (lines.loc[lines['selected'], 'obs_wave_min'], 'selected', COLORS['ref']),
                      (lines.loc[lines['telluric'], 'obs_wave_min'], 'telluric (removed)', COLORS['tell']),
                      (lines.loc[lines['selected'] & lines['blend'], 'obs_wave_min'], 'selected, blends', '#7b4fa0'),
                      (reference['wave'], 'reference mask', COLORS['syn']),
                      (lines.loc[lines['in_reference'], 'obs_wave_min'], 'in the reference mask', COLORS['obs'])):
        ax.step(mid, h(x), where='mid', color=c, label=lab)
    ax.set_xlabel(r'Wavelength [$\AA$]'); ax.set_ylabel('lines per %.0f $\\AA$' % step)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _style(ax)
    return fig


def telluric_fit(wave, observed, synthetic, model, bands):
    """observed / synthetic and the telluric model in the centre (6 A) of every band."""
    fig, axes = plt.subplots(2, int(np.ceil(len(bands) / 2)), figsize=(6 * np.ceil(len(bands) / 2), 7), squeeze=False)
    for ax, (band, (mol, region, _)) in zip(axes.ravel(), bands.items()):
        mid = 0.5 * (region[0] + region[1])
        k = np.abs(wave - mid) < 3
        kt = np.abs(model[0] - mid) < 3
        ax.plot(wave[k], observed[k] / synthetic[k], 'k', lw=0.8, label='observed / synthetic')
        ax.plot(model[0][kt], model[1][kt], color=COLORS['tell'], lw=1, ls='--', label='telluric model')
        ax.set_title(band); ax.set_xlabel(r'Wavelength [$\AA$]')
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    return fig


def telluric_lines(lines, threshold):
    """Telluric depth in the window of every common line, the threshold, and the lines removed."""
    fig, ax = plt.subplots(figsize=(15, 4))
    t = lines['telluric']
    ax.plot(lines.loc[~t, 'obs_wave_min'], lines.loc[~t, 'telluric_depth'] + 1e-6, '.', ms=2, color=COLORS['grey'], label='kept')
    ax.plot(lines.loc[t, 'obs_wave_min'], lines.loc[t, 'telluric_depth'], '.', ms=3, color=COLORS['tell'],
            label='telluric (removed): %d' % t.sum())
    ax.axhline(threshold, color='k', lw=0.8, ls='--')
    ax.set_yscale('log'); ax.set_ylim(1e-5, 1.2)
    ax.set_xlabel(r'Wavelength [$\AA$]'); ax.set_ylabel('telluric depth in the window')
    ax.legend(frameon=False)
    _style(ax)
    return fig


def ccfs_and_bisectors(rv, curves, title=''):
    """CCFs (normalised, centred on their RV) and bisectors, as absorption lines (core at the bottom, continuum at the
    top): curves {label: (ccf, params of ccf_params)}."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, (label, (c, p)) in enumerate(curves.items()):
        cn = (c - c.min()) / (c.max() - c.min())
        axes[0].plot((rv - p['rv']) / 1e3, 1 - cn, color=SERIES[k % len(SERIES)], lw=1.2, label=label)
        axes[1].plot(p['x_bis'] - p['rv'], 1 - p['y_bis'], color=SERIES[k % len(SERIES)], lw=1.2,
                     label='%s: BIS %.1f m/s' % (label, p['bis']))
    axes[0].set_xlabel('RV - RV of the CCF [km/s]'); axes[0].set_ylabel('normalised CCF flux')
    axes[1].set_xlabel('bisector - RV of the CCF [m/s]'); axes[1].set_ylabel('normalised CCF flux')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    for ax in axes:
        _style(ax)
    return fig


def timeseries(table, masks, keys=('rv', 'fwhm', 'bis'), t=None, extra=None, xlabel='Time [min]'):
    """Indicators of every mask minus their median against time; extra {label: column prefix} adds e.g. the DRS."""
    t = (table['bjd'] - table['bjd'].min()) * 1440 if t is None else t
    fig, axes = plt.subplots(len(keys), 1, figsize=(13, 3.2 * len(keys)), sharex=True)
    for ax, key in zip(np.atleast_1d(axes), keys):
        for k, name in enumerate(list(masks) + list((extra or {}).keys())):
            col = '%s_%s' % (name, key) if name in masks else '%s%s' % (extra[name], key)
            if col not in table:
                continue
            x = table[col]
            ax.plot(t, x - x.median(), '.-', ms=3, lw=0.6, color=SERIES[k % len(SERIES)],
                    label='%s (std %.2f)' % (name, np.std(x)))
        ax.set_ylabel('%s - median [m/s]' % key.upper())
        ax.legend(frameon=False, fontsize=7, ncol=3)
        _style(ax)
    np.atleast_1d(axes)[-1].set_xlabel(xlabel)
    return fig


def template_iterations(hist):
    """Scatter of the indicators of every mask against the iteration of the template: hist {mask: history}."""
    keys = ('rv std', 'fwhm std', 'bis std', 'template change')
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.8))
    for k, (m, h) in enumerate(hist.items()):
        it = [r['iteration'] for r in h]
        for ax, key in zip(axes, keys):
            ax.plot(it, [r[key] for r in h], 'o-', ms=4, color=SERIES[k % len(SERIES)], label=m)
    for ax, key in zip(axes, keys):
        ax.set_xlabel('iteration'); ax.set_title(key)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------------------------------------------------
def flag_scatter(ax, x, y, category, styles, ms=2.5, alpha=0.5, label=True):
    """x against y with one style per category (flags.first_flag): styles = {category: dict(color, marker, label)},
    drawn in the order of styles (first: bottom). Categories not in styles are not drawn."""
    x, y, category = np.asarray(x), np.asarray(y), np.asarray(category)
    for cat, st in styles.items():
        k = category == cat
        if not k.any():
            continue
        lab = '%s (%d)' % (st.get('label', cat), k.sum()) if label else None
        ax.plot(x[k], y[k], st.get('marker', '.'), ms=st.get('ms', ms), color=st['color'], alpha=st.get('alpha', alpha),
                label=lab, mfc=st.get('mfc', st['color']), mew=st.get('mew', 0.8), zorder=st.get('zorder', 2))
    return ax


# ---------------------------------------------------------------------------------------------------------------------
# extract_mask / compare_masks (any number of spectra)
# ---------------------------------------------------------------------------------------------------------------------
def extraction_zoom(result, spectra, window=(5000.0, 5010.0), vald=None):
    """Spectra {name: (wave, flux, ...)} in a wavelength window with the lines of an extract_mask result: in the mask
    'clean' (triangle), blend (purple triangle), removed by another flag (cross), and the line list (depth bars)."""
    fig, ax = plt.subplots(figsize=(15, 4))
    for k, (label, s) in enumerate(spectra.items()):
        w, f = np.asarray(s[0]), np.asarray(s[1])
        q = (w > window[0]) & (w < window[1])
        ax.plot(w[q], f[q], lw=1, color=SERIES[k % len(SERIES)], label=label)
    if vald is not None:
        v = vald[(vald['wave'] > window[0]) & (vald['wave'] < window[1])]
        ax.vlines(v['wave'], 1 - v['depth_vald'], 1, color='g', alpha=0.25, lw=0.8, label='line list')
    t = result.lines
    t = t[(t['wave'] > window[0]) & (t['wave'] < window[1])]
    from .extract import _label
    fmin = t['%s_flux_min' % _label(result.primary)]
    other = t[['telluric', 'shallow']].any(axis=1) | (t['no_vald'] & bool(result.config.get('require_vald')))
    for sel, mk, c, lab in ((~other & ~t['blend'], 'v', 'k', 'mask line'), (~other & t['blend'], 'v', '#7b4fa0', 'blend'),
                            (other, 'x', COLORS['tell'], 'removed (telluric / shallow / no line-list line)')):
        if sel.any():
            ax.plot(t.loc[sel, 'wave'], fmin[sel] - 0.03, mk, ms=5, color=c, label=lab)
    ax.set_xlim(*window)
    ax.set_xlabel(r'Wavelength [$\AA$]'); ax.set_ylabel('Normalised flux')
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _style(ax)
    return fig


def mask_comparison(comparison, names=('mask', 'reference')):
    """compare_masks output: matched weights (log-log), mask - reference wavelengths [m/s], and the weights of the
    matched lines along the wavelength."""
    m = comparison['matched']
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    wa, wb = m['weight'] * len(m), m['ref_weight'] * len(m)
    pos = (wa > 0) & (wb > 0)
    axes[0].loglog(wb[pos], wa[pos], '.', ms=2, alpha=0.4, color=SERIES[0])
    lim = [min(wa[pos].min(), wb[pos].min()), max(wa.max(), wb.max())]
    axes[0].plot(lim, lim, 'k--', lw=0.8)
    axes[0].set_xlabel('%s weight (x N matched)' % names[1]); axes[0].set_ylabel('%s weight (x N matched)' % names[0])
    axes[0].set_title('r = %.3f, log r = %.3f' % (comparison['counts']['weight r'], comparison['counts']['log weight r']),
                      fontsize=10)
    lim_v = max(500.0, np.percentile(np.abs(m['dv']), 99)) if len(m) else 1000.0
    axes[1].hist(m['dv'], bins=np.linspace(-lim_v, lim_v, 101), color=SERIES[1])
    axes[1].set_xlabel('%s - %s [m/s] (shift %.0f m/s removed)' % (names[0], names[1], comparison['shift']))
    axes[1].set_ylabel('lines')
    axes[2].plot(m['ref_wave'], np.log10(wa / np.where(wb > 0, wb, np.nan)), '.', ms=2, alpha=0.4, color=SERIES[2])
    axes[2].axhline(0, color='k', lw=0.8)
    axes[2].set_xlabel(r'Wavelength [$\AA$]'); axes[2].set_ylabel('log10 (%s / %s weight)' % names)
    for ax in axes:
        _style(ax)
    return fig
