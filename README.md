# dupont

Developed by Sophie Stucki, in collaboration with Khaled Al Moulla.

CCF line masks built from an observed spectrum and a synthetic one. A line is kept when it is seen in both (for
example the Sun observed with ESPRESSO through PoET and a MURaM simulation), so the same mask can be used on
observations and on simulations. `extract_mask` (below) does the same for one spectrum or any number of spectra.

## Steps

1. **Observed reference.** `stack_s1d` averages S1D spectra of one night and propagates their errors.
   `flatten` divides the flux and the error by a continuum (SNT).
2. **Synthetic at the instrument resolution.** `degrade_resolution` convolves the synthetic spectrum with a Gaussian
   of constant velocity width, c / R. For ESPRESSO UHR, R = 190 000.
3. **Line detection.** `detect_lines` finds every local minimum with a given prominence. It takes the first maximum on
   each side as the window and fits a Gaussian to the core to get the minimum.
4. **Common lines.** `pair_lines` pairs the minima of the two spectra one to one, closest first, within a velocity
   tolerance after removing the global shift. `shared_window` gives both lines the same pixels.
5. **Identification.** `identify` labels each line with the VALD line in its window that best matches in wavelength
   and depth: distance in resolution elements (λ / R) plus |ln(k × VALD depth / depth)|, k = median depth / VALD depth
   (VALD depths are unbroadened). The number of VALD lines in the window flags blends.
6. **Weights.** `rv_weights` computes the RV information of each line (Bouchy et al. 2001):
   eq. 8, W = Σ λ² (∂A₀/∂λ)² / (A₀ + σ_D²) over the line window, with A₀ the flattened flux of each spectrum and σ_D
   the detector noise (0 by default): the weights depend only on the spectrum, not on the S/N of the observation.
   With the measured error, W = Σ λ² (dF/dλ)² / σ² gives the photon-noise RV error of each line in the observation
   (`sigma_rv_stack`).
7. **Tellurics.** `fit_tellurics` fits HITRAN H₂O and O₂ absorption to observed / synthetic. `telluric_depth` then
   gives the deepest telluric absorption inside each line window, widened by the range of Earth velocities. A line
   deeper than the threshold is dropped.
8. **Reference mask.** `match_reference` cross-matches a reference mask (e.g. the ESPRESSO one) one to one.
9. **CCFs.** `ccf` and `ccf_params` give RV, FWHM, contrast and BIS. `ccf_timeseries`, `summary` and
   `differential_scatter` compare masks over many spectra.

`build_mask` runs steps 3 to 8 with a `MaskConfig`, and `save` writes the masks together with the parameters that
made them.

## Any spectra: `extract_mask`

`extract_mask` does the same with one spectrum or several, of any kind (observation + simulation, two simulations,
two instruments, ...). The lines found in every spectrum are kept. The first spectrum (or `primary=`) gives the mask
wavelengths and weights.

```python
res = lm.extract_mask({'PoET': (w, f, e), 'MURaM': (w2, f2)}, lm.ExtractConfig(prominence=0.05),
                      vald=vald, telluric=(w_t, t))            # vald and telluric are optional
res.masks['all'], res.masks['clean']                           # DataFrames (wave, weight)
res.mask(exclude=('telluric', 'shallow', 'blend'), weight='MURaM')   # another selection, no new extraction
esp = lm.read_mask(lm.mask_path('ESPRESSO_SUN.mas'))
cmp = lm.compare_masks(res.masks['all'], esp, names=('own', 'ESPRESSO'))
table, curves = lm.compare_ccfs({'PoET': (w, f)}, {'own': res.masks['all'], 'ESPRESSO': esp})
```
The flags are `telluric`, `shallow` (`min_depth`), `no_vald` and `blend`. `read_mask` reads csv, FITS and `.mas`;
`write_mask` writes csv or `.mas`. With the `build_mask` defaults, `extract_mask` gives the same mask as `build_mask`.

## Preparing spectra: `prepare`

`degrade_resolution` (gaussian IP, c / R), `resample` (flux-conserving, onto instrument pixels), `velocity_grid`,
`rotational_broadening` (other stars), `add_noise`, `to_rest` / `measure_rv` (frame of a mask), `crop`,
`vacuum_to_air`, `doppler_shift`.

Synthetic spectra at several µ angles (MURaM, PoET, ...) can be integrated over the disc with SunSim in its
high-resolution spectroscopic mode, `sunsim.flux_grid(..., mode='spectroscopy', stellar_rotation=True)`:
`read_mu_spectra` + `disc_integrate` (solar differential rotation, 60 rings by default; needs `sunsim`).

## Masks shipped with dupont

`lm.mask_path(name)` gives the path of a mask in `src/dupont/data/masks`:

| file | content |
|---|---|
| `ESPRESSO_SUN.csv` | ESPRESSO solar mask (ESPRESSO DRS): `wave` [Å, air], `cont` (line contrast) |
| `ESPRESSO_SUN.mas` | the same lines, weight = contrast² normalised to 1 (SunSim format) |
| `PoET_noflag.mas` | mask built with dupont from PoET (ESPRESSO, 15 May 2026) and MURaM, lines without any flag |

## Tutorial

`tutorial.ipynb` builds the PoET × MURaM mask step by step: stack and flattening of the PoET S1D, MURaM disc
integration and resolution, tellurics, line detection and pairing, presence in every exposure, VALD identification,
weights, ESPRESSO match, flags, CCFs and the RV time series, then the same mask with `extract_mask`.

It needs data that are not shipped, found with the environment variables below (or, if `sunsim` is installed, with
SunSim's `local_paths.ini`):

| variable | content |
|---|---|
| `DUPONT_POET_DATA` | `S1D_A/*_S1D_A.fits`; `output/SUN_LINELIST_VALD_2.csv` (VALD), `output/SUN_SPECTRUM_MURAM_DISC_N60_R50.csv`, `output/mask_flags/TELL_SPECTRUM_HITRAN_JOINT.csv`, `output/mask_flags/telluric_joint_fits.csv` |
| `DUPONT_MURAM_SPECTRA` | `av_solar_ssd_200G_10_mu_angles_PHOENIX_high_res_grid.npy` (MURaM µ grid) |

## Install

```
pip install -e .            # numpy, scipy, pandas, astropy, matplotlib
pip install -e ".[disc]"    # + sunsim, for disc_integrate
pip install -e ".[all]"     # + sunsim, loguru (SNT), hitran-api, pytest
```
Flattening needs [SNT](https://github.com/iastro-pt/Spectra-Normalization-Tool). Computing new HITRAN optical depths
needs `hapi`; a cached grid can be read without it.

## Tests

```
pytest tests
```
The tests inject lines into synthetic spectra and check that the pipeline recovers them: positions, depths,
resolution, pairing with a shift, weights, the one-to-one match, telluric windows, CCF RV and the full mask.

## Credits and license

Written by Sophie Stucki, in collaboration with Khaled Al Moulla. Released under the MIT license (see `LICENSE`).
