"""Line table with flags: every threshold of the mask is a flag (True = the line fails that test), nothing is removed.

The table has one row per line common to the observed and synthetic spectra; the masks are then selections of flags
(select), so that every threshold can be followed in the plots and changed afterwards.
"""
import numpy as np
import pandas as pd

from .spectra import C_LIGHT


def presence_in_windows(spectra, wave_left, wave_right, prominence=0.025, shift=0.0):
    """Number of spectra with a line inside every window.

    spectra: list of (wave, flux) flattened; wave_left, wave_right: the window of every line [A] (in the frame of the
    stack the lines come from); shift: velocity of the spectra relative to that frame [m/s]. In every spectrum the
    minima are found with scipy find_peaks on -flux with this prominence (minimum pixel, no fit); a line is present when
    at least one minimum falls inside its window (wave_left < minimum < wave_right). Returns an int array (one per
    line)."""
    from .lines import detect_lines
    lo = np.asarray(wave_left, dtype=np.float64) * (1 + shift / C_LIGHT)
    hi = np.asarray(wave_right, dtype=np.float64) * (1 + shift / C_LIGHT)
    n = np.zeros(len(lo), dtype=int)
    for w, f in spectra:
        d = np.sort(detect_lines(w, f, prominence=prominence, center='pixel')['wave_min'].values)
        n += np.searchsorted(d, hi, side='left') > np.searchsorted(d, lo, side='right')
    return n


def select(table, flags):
    """Lines with none of these flags (boolean Series)."""
    return ~table[list(flags)].any(axis=1)


def first_flag(table, order, none='no flag'):
    """Category of every line for the plots: the first flag of `order` the line has, `none` if it has none."""
    out = pd.Series(none, index=table.index, dtype=object)
    for f in reversed(list(order)):
        out[table[f].values] = f
    return out


def summary(table, flags, weight='obs_weight'):
    """For every flag: number of lines, their share of the summed weight, and how many of them have no other flag of
    the list. Last row: lines with no flag."""
    flags = list(flags)
    tot = table[weight].sum()
    rows = {}
    for f in flags:
        k = table[f].values
        others = table[[g for g in flags if g != f]].any(axis=1).values
        rows[f] = {'lines': int(k.sum()), 'weight share [%]': 100 * table.loc[k, weight].sum() / tot,
                   'only this flag': int((k & ~others).sum())}
    k = select(table, flags).values
    rows['no flag'] = {'lines': int(k.sum()), 'weight share [%]': 100 * table.loc[k, weight].sum() / tot,
                       'only this flag': np.nan}
    return pd.DataFrame(rows).T


def overlaps(table, flags):
    """Number of lines with every pair of flags (diagonal: lines with the flag)."""
    flags = list(flags)
    m = table[flags].values.astype(int)
    return pd.DataFrame(m.T @ m, index=flags, columns=flags)
