# Response-regime diagnostics

The PSMAP response is represented exactly as

\[r(z,\varphi)=a(z)+\operatorname{Re}[q(z)e^{i\varphi}].\]

The repository maps are regular 21×21×21×21 grids in `(x0,y0,vx0,vy0)` (SI
units, each transverse coordinate spanning ±0.6 mm). Phase is not a grid axis.
Each atom/port record contains two real path amplitudes and their phase shift,
so `a=amp0²+amp1²` and `q=2 amp0 amp1 exp(i phase_shift)` for interfering
paths. Consequently the first-harmonic residual is zero up to floating-point
roundoff; no higher harmonics are present in this representation. Z0 and Z100
are separate maps for the two baseline locations. Existing simulation uses
`aispy.psmap.load_psmap` and Catmull–Rom interpolation; the diagnostics reader
loads the identical numeric HDF5 fields while omitting the very large path-name
strings. Operational projection uses deterministic native-grid trapezoid
quadrature and the established 3.8 s ballistic detection time.

## Intrinsic versus operational statements

Intrinsic diagnostics use equal node weight and require no cloud: the
dimensionless ballistic generator violation, within-final-bin variance,
distance from `q_b = gamma a_b`, phase coherence, and the two singular values
of the image-space phase orbit. They say what symmetries the response permits,
not whether an experiment can resolve the permitted structure.

Operational diagnostics require the Gaussian cloud widths, ROI/pixels, atom
number and nuisance scale. They include Poisson count/shape Fisher information,
phase uncertainty, and nuisance information after Poisson-metric projection of
the phase tangent. The hidden mean mode changes `mu_vx` by 10 µm/s and `mu_x`
by `-t_det` times that amount, preserving final mean position. Its final-bin
mass derivative is explicitly nulled before measuring response covariance, so
finite grid truncation is not mistaken for response information.

The information identity used in the plots is

`I_image = I_count + I_shape`.

Count-only compression is exact only when normalized shape is ancillary, in
particular when every pixel phasor obeys `q_b = gamma a_b`. Otherwise a useful
compression must retain phase/shape information and its covariance.

## Current-map result (nominal 10^6 atoms)

With 100 µm RMS initial position and velocity widths, 24×24 pixels across five
final-cloud RMS widths, and the 10 µm/s hidden-mode scale, Z0 is operationally
Regime I/mixed and Z100 is Regime II. Both intrinsically break ballistic-fibre
symmetry (generator violations 0.222 and 0.285), but only 1.89% and 2.73% of the
phasor variation is within the approximate fibres. Z0's hidden-mode SNR is
0.72; Z100's is 1.42. Amplifying the existing fibre-breaking component reaches
SNR one at alpha=1.50 for Z0 and alpha=0.75 for Z100.

Neither map is carrier-like. Their cloud-projected orbit axis ratios are 0.0136
and 0.0129 and their minimum/median phase-information ratios are about
4.5e-4. Shape contains about 2.9% and 2.7% of phase information on average, so
it does directly measure some phase, but conditioning varies dramatically over
the orbit. Under the configured Regime-III criteria, the first tested shear
meeting the threshold is 2665 rad/m (about 0.165 fringes per final RMS width)
for both maps. The scan is discrete; this is a bracketed threshold, not a
high-precision root.

For the current unsheared maps, posterior propagation or amortized joint
`(eta,phi)` inference is preferable to direct per-shot phase-and-summary
regression. Z0 is close to the I/II boundary; Z100 has resolvable hidden-mode
response information and needs image-level/joint inference. Added shear makes
direct phase-and-summary in-situ inference plausible. In real data, nuisance
posterior uncertainty must be propagated. Simulation point estimates may be
assessed by repeated runs. Learning physical `eta` is interpretable and
transferable; learning a bias can be simpler but that bias is a valid label
only from simulator truth or calibrated injected truth. A learned joint
posterior or compressed likelihood retains nuisance marginalization most
faithfully.

Machine-readable results and all figures are in
`results/response_regime_diagnostics/`. Thresholds and numerical configuration
are recorded in `summary.json`; continuous metrics should always accompany the
regime label.
