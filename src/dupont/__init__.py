"""dupont: CCF line masks from one or more spectra (extract_mask), or from an observed and a synthetic spectrum with
the PoET / MURaM settings (build_mask).

Lines are kept when they are detected in both spectra (e.g. the Sun observed with ESPRESSO and a 3D MHD simulation),
identified in a line list (VALD), weighted by their RV information (Bouchy et al. 2001, eq. 8, photon noise of
each spectrum's own flux), freed from telluric absorption with a fitted telluric model, and cross-matched with a reference mask.
"""
__author__ = 'Sophie Stucki (in collaboration with Khaled Al Moulla)'
__license__ = 'MIT'
from .spectra import (C_LIGHT, ESPRESSO_RESOLUTION, FLATTEN, common_grid, continuum_snt, degrade_resolution,
                      doppler_shift, flatten, load_spectrum, read_s1d, save_spectrum, stack_s1d, vacuum_to_air)
from .lines import detect_lines, identify, load_vald, refine_minimum, vald_score
from .pairing import match_reference, pair_lines, rv_weights, shared_window
from .tellurics import (add_trace_gases, fetch_hitran, fit_quality, fit_tellurics, fit_tellurics_joint, hitran_tau_grid,
                        telluric_depth, telluric_model, telluric_model_joint)
from .ccf import RV_GRID, ccf, ccf_params, photon_rv_error
from .timeseries import ccf_timeseries, differential_scatter, line_presence, summary
from .templatefit import template_fit, template_fit_iterative, template_reference
from .pipeline import MaskConfig, build_mask, load_reference_mask, save
from .extract import (FLAGS, ExtractConfig, MaskResult, common_grid_many, compare_ccfs, compare_masks, extract_mask,
                      read_mask, write_mask)
from .prepare import (add_noise, crop, disc_integrate, measure_rv, read_mu_spectra, resample, rotational_broadening,
                      to_rest, velocity_grid)
from .data import MASKS, mask_path

from . import flags  # noqa: E402
from . import prepare  # noqa: E402
from . import plots  # noqa: E402  (matplotlib, imported with the package)

__version__ = '0.1.0'
