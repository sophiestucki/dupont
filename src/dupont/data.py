"""Masks shipped with dupont (src/dupont/data/masks).

ESPRESSO_SUN.csv  ESPRESSO solar mask of the ESPRESSO DRS: wave [Å, air], cont (line contrast)
ESPRESSO_SUN.mas  the same lines, weight = contrast^2 normalised to 1 (SunSim format: wave, wave, weight)
PoET_noflag.mas   mask built with dupont from PoET (ESPRESSO, 15 May 2026) and MURaM, lines without any flag
"""
from importlib.resources import files
from pathlib import Path

MASKS = ('ESPRESSO_SUN.csv', 'ESPRESSO_SUN.mas', 'PoET_noflag.mas')


def mask_path(name):
    """Path of a shipped mask (see MASKS), e.g. read_mask(mask_path('ESPRESSO_SUN.mas'))."""
    if name not in MASKS:
        raise ValueError('unknown mask %r, shipped masks: %s' % (name, ', '.join(MASKS)))
    return Path(str(files('dupont') / 'data' / 'masks' / name))
