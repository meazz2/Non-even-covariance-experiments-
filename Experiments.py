#!/usr/bin/env python
"""Figures 1-7 and parameter recovery simulations.

Python >=3.8; NumPy, SciPy >=1.10, pandas >=2.0, Matplotlib, threadpoolctl.
Standalone: no notebook or auxiliary Python files are needed.

Example:
  python corrected_paper_experiments.py --output paper_experiment_results --workers 4
Resume interrupted runs by adding --resume. Defaults: 200 datasets/model,
one bivariate field realization/dataset, 64 sites. No truth-centred starts,
eigenvalue clipping, hidden nugget, or outlier deletion.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
for _thread_variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_thread_variable,'1')
# Python 3.8+ on Windows needs the selected conda environment's DLL directory.
_native_dll_handles=[]
if sys.platform=='win32':
    _conda_native=Path(sys.prefix)/'Library/bin'
    if _conda_native.is_dir():
        os.environ['PATH']=str(_conda_native)+os.pathsep+os.environ.get('PATH','')
        if hasattr(os,'add_dll_directory'):
            _native_dll_handles.append(os.add_dll_directory(str(_conda_native)))
import matplotlib
matplotlib.use('Agg')


# validity.py
"""Continuous-frequency spectral bounds for the corrected covariance models.

The pure-component calculation maximizes the squared cross/marginal spectral
ratio over every frequency. Matérn stationary points are roots of a polynomial
of degree at most three; this is not a frequency-grid test. Analytic endpoint
limits are included. Equal-scale Matérn and Gaussian cases have closed forms.

Floating-point polynomial solutions are checked for residuals, reconstruction,
and clustered roots. Unreliable roots trigger an analytic conservative upper
bound instead. The returned finite bounds include a relative 1e-10 upward
margin; they are numerical spectral bounds, not interval-arithmetic proofs.
The mixed bound is sufficient and may restrict the full admissible family.
"""

import math
from decimal import Decimal, localcontext

import numpy as np
from scipy.special import gammaln


_LOG_BOUND_MARGIN = math.log1p(1.0e-10)


def _positive_number(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("{} must be positive and finite".format(name))
    return value


def _log_b(nu, d):
    # Gamma(nu+1)/Gamma(nu)=nu avoids cancellation in the planar case.
    return math.log(nu) if d == 2 else float(gammaln(nu + d / 2.0) - gammaln(nu))


def _log_beta_max(k, delta):
    """log max_{0<=t<=1} t**k*(1-t)**delta, including boundaries."""
    if k == 0 or delta == 0.0:
        return 0.0
    total = k + delta
    return k * math.log(k / total) + delta * math.log(delta / total)


def _matern_log_sup(a11, a22, ac, nu11, nu22, nuc, k, d):
    # A negative exponent deficit makes the spectral ratio unbounded.
    delta = math.fsum([2.0 * nuc, -nu11, -nu22, -float(k)])
    if delta < 0.0:
        return math.inf
    log_r1 = 2.0 * (math.log(a11) - math.log(ac))
    log_r2 = 2.0 * (math.log(a22) - math.log(ac))
    alpha, beta = nu11 + d / 2.0, nu22 + d / 2.0
    log_constant = (2.0 * _log_b(nuc, d) - _log_b(nu11, d)
                    - _log_b(nu22, d) - nu11 * log_r1 - nu22 * log_r2)
    log_beta_max = _log_beta_max(k, delta)

    # With t=x/(ac**2+x), each linear term lies between ri and one.
    # Maximizing factors separately therefore supplies an analytic fallback.
    fallback = (log_constant + log_beta_max
                + alpha * max(log_r1, 0.0) + beta * max(log_r2, 0.0))
    if a11 == ac and a22 == ac:
        return log_constant + log_beta_max + _LOG_BOUND_MARGIN

    # Dimensionless y=x/ac**2 prevents units from conditioning the polynomial.
    # Extreme scale ratios use the overflow-safe logarithmic fallback.
    if max(abs(log_r1), abs(log_r2)) > 250.0:
        return fallback + _LOG_BOUND_MARGIN
    r1, r2 = math.exp(log_r1), math.exp(log_r2)
    gamma = 2.0 * nuc + d
    if k:
        coefficients = np.array([
            float(k) * r1 * r2,
            k * (r1 * r2 + r1 + r2) + alpha * r2 + beta * r1 - gamma * r1 * r2,
            k * (r1 + r2 + 1.0) + alpha * (r2 + 1.0)
            + beta * (r1 + 1.0) - gamma * (r1 + r2),
            -delta,
        ], dtype=float)
    else:
        coefficients = np.array([
            alpha * r2 + beta * r1 - gamma * r1 * r2,
            alpha * (r2 + 1.0) + beta * (r1 + 1.0) - gamma * (r1 + r2),
            -delta,
        ], dtype=float)
    # Drop only exact zero leading coefficients, never a small tail deficit.
    while len(coefficients) > 1 and coefficients[-1] == 0.0:
        coefficients = coefficients[:-1]
    coefficient_scale = np.max(np.abs(coefficients))
    if not np.all(np.isfinite(coefficients)):
        return fallback + _LOG_BOUND_MARGIN

    # Evaluate in log(y); logaddexp handles arbitrarily large positive roots.
    def log_ratio(log_y):
        log_t = -float(np.logaddexp(0.0, -log_y))
        log_one_minus_t = -float(np.logaddexp(0.0, log_y))
        return (log_constant + k * log_t + delta * log_one_minus_t
                + alpha * float(np.logaddexp(log_r1 + log_one_minus_t, log_t))
                + beta * float(np.logaddexp(log_r2 + log_one_minus_t, log_t)))

    values = []
    if k == 0:
        values.append(log_constant + alpha * log_r1 + beta * log_r2)
    if delta == 0.0:
        values.append(log_constant)
    # An identically zero derivative means a constant ratio.
    if coefficient_scale == 0.0:
        return max(values) + _LOG_BOUND_MARGIN
    if len(coefficients) > 1:
        normalized = coefficients / coefficient_scale
        try:
            with np.errstate(all="ignore"):
                roots = np.roots(normalized[::-1])
                reconstructed = np.poly(roots) * normalized[-1]
                if (not np.all(np.isfinite(roots))
                        or not np.all(np.isfinite(reconstructed))
                        or not np.allclose(reconstructed, normalized[::-1],
                                           rtol=1.0e-7, atol=1.0e-10)):
                    return fallback + _LOG_BOUND_MARGIN
                for index, root in enumerate(roots):
                    size = max(1.0, abs(root))
                    # Coalescing roots are where companion eigenvalues can
                    # obscure real stationary points; do not trust that case.
                    for other in roots[index + 1:]:
                        if abs(root - other) < 1.0e-6 * max(size, abs(other)):
                            return fallback + _LOG_BOUND_MARGIN
                    residual = abs(np.polynomial.polynomial.polyval(root, normalized))
                    residual_scale = np.polynomial.polynomial.polyval(abs(root), np.abs(normalized))
                    if (not np.isfinite(residual) or not np.isfinite(residual_scale)
                            or residual > 1.0e-7 * residual_scale):
                        return fallback + _LOG_BOUND_MARGIN
                    if abs(root.imag) <= 1.0e-10 * size:
                        if root.real > 0.0:
                            values.append(log_ratio(math.log(float(root.real))))
                    elif abs(root.imag) < 1.0e-6 * size:
                        return fallback + _LOG_BOUND_MARGIN
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            return fallback + _LOG_BOUND_MARGIN
    if not values or not np.all(np.isfinite(values)):
        return fallback + _LOG_BOUND_MARGIN
    maximum = max(values)
    # A candidate exceeding a proven factorwise bound flags numerical trouble.
    if maximum > fallback + 1.0e-7 * max(1.0, abs(fallback)):
        return fallback + _LOG_BOUND_MARGIN
    return maximum + _LOG_BOUND_MARGIN


def log_component_sup(family, a11, a22, ac, nu11=None, nu22=None,
                      nuc=None, p=1, odd=False, d=2):
    """Return log of the continuous-frequency squared spectral-ratio bound.

    Gaussian odd multiplier: u**p. Matérn odd multiplier: (u/ac)**p.
    The normalized angular factor has supremum one. Gaussian even requires
    2*ac**2 >= a11**2+a22**2; odd requires strict inequality. Matérn requires
    2*nuc >= nu11+nu22+(p if odd else 0). Failed restrictions return infinity.
    Invalid marginal scales/smoothness raise ValueError. A zero cross-amplitude
    or an inactive component should be omitted before calling this function.
    """
    if int(d) != d or d < 2:
        raise ValueError("d must be an integer at least two")
    if odd and (int(p) != p or p <= 0 or int(p) % 2 != 1):
        raise ValueError("p must be a positive odd integer")
    a11 = _positive_number(a11, "a11")
    a22 = _positive_number(a22, "a22")
    ac = _positive_number(ac, "ac")
    k = int(p) if odd else 0
    family = str(family).lower()
    if family == "gaussian":
        # Scale first to avoid squaring very large physical scales.
        largest = max(a11, a22, ac)
        scaled_delta = math.fsum([2.0 * (ac / largest) ** 2,
                                 -(a11 / largest) ** 2, -(a22 / largest) ** 2])
        log_delta = None
        if abs(scaled_delta) < 1.0e-12:
            # Close to the equality boundary, a rounded cancellation must not
            # turn a negative Delta into an apparently admissible positive one.
            with localcontext() as context:
                context.prec = 80
                exact_delta = (2 * Decimal.from_float(ac) ** 2
                               - Decimal.from_float(a11) ** 2
                               - Decimal.from_float(a22) ** 2)
                if exact_delta < 0 or (odd and exact_delta == 0):
                    return math.inf
                if exact_delta > 0:
                    log_delta = float(exact_delta.ln())
                    scaled_delta = 1.0  # The sign has been established above.
                else:
                    scaled_delta = 0.0
        if scaled_delta < 0.0 or (odd and scaled_delta == 0.0):
            return math.inf
        answer = 2.0 * d * math.log(ac) - d * (math.log(a11) + math.log(a22))
        if odd:
            if log_delta is None:
                log_delta = math.log(scaled_delta) + 2.0 * math.log(largest)
            answer += k * (math.log(4.0 * k) - log_delta - 1.0)
        return answer + _LOG_BOUND_MARGIN
    if family == "matern":
        nu11 = _positive_number(nu11, "nu11")
        nu22 = _positive_number(nu22, "nu22")
        nuc = _positive_number(nuc, "nuc")
        if odd and nuc <= p / 2.0:
            return math.inf
        return _matern_log_sup(a11, a22, ac, nu11, nu22, nuc, k, d)
    raise ValueError("family must be 'gaussian' or 'matern'")


def mixed_log_sup_bound(log_even, log_odd, theta):
    """Log of cos(theta)^2*M_even + sin(theta)^2*M_odd.

    This is a sufficient upper bound on the mixed spectral supremum. It equals
    the component bound for a pure angle. Inactive terms are omitted before
    inspecting them, so an unused infinity does not cause 0*infinity or NaN.
    Exact multiples of pi/2 are recognized without an arbitrary angle cutoff.
    """
    theta = float(theta)
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    phase = math.remainder(theta, math.pi)
    if phase == 0.0:
        return float(log_even)
    if abs(phase) == math.pi / 2.0:
        return float(log_odd)
    even = float(log_even) + 2.0 * math.log(abs(math.cos(phase)))
    odd = float(log_odd) + 2.0 * math.log(abs(math.sin(phase)))
    return float(np.logaddexp(even, odd))


# model_engine.py


from dataclasses import dataclass, field, asdict
from functools import lru_cache
from contextlib import nullcontext
import math
import time
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import eval_chebyt, eval_gegenbauer, gammaln, kve
from scipy.stats import qmc


@dataclass(frozen=True)
class ModelSpec:
    family: str
    kind: str
    p: int = 1

    def __post_init__(self):
        if self.family not in ('G', 'M') or self.kind not in ('even', 'odd', 'mixed'):
            raise ValueError('Use family G/M and kind even/odd/mixed.')
        if isinstance(self.p,(bool,np.bool_)) or not isinstance(self.p, (int, np.integer)) or self.p < 1 or self.p % 2 != 1:
            raise ValueError('p must be a positive odd integer.')

    @property
    def name(self):
        return self.family + '_' + self.kind

    @property
    def has_even(self):
        return self.kind != 'odd'

    @property
    def has_odd(self):
        return self.kind != 'even'


@dataclass
class FitConfig:
    seed: int = 1729
    n_candidates: int = 48
    n_starts: int = 3
    maxiter: int = 160
    ftol: float = 1e-8
    # Lengths are in the common numerical reference unit; None derives bounds
    # from the minimum spacing and domain diameter, identically for all models.
    length_bounds: tuple = None
    marginal_nu_bounds: tuple = (0.15, 8.0)
    cross_nu_gap_max: float = 8.0
    gaussian_gap_max: float = 20.0
    gaussian_odd_gap_min: float = 1e-3
    fraction_limit: float = 0.995
    mean_mode: str = 'zero'       # 'zero' matches the original; 'profile' fits 2 means
    marginal_sds: tuple = (1.0, 1.0)
    nugget_variance: float = 0.0  # explicit, fixed observation-noise variance
    bic_count: str = 'scalar'    # 2*N*R, or 'locations' (N*R), or 'replicates' (R)
    blas_threads: int = 1
    fixed_smoothness: dict = field(default_factory=dict)
    # fixed_smoothness keys: nu11, nu22, nu_even, nu_odd. Cross values must
    # satisfy the smoothness tail restriction for each fitted p.

    def validate(self):
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v < 1
               for v in (self.n_candidates,self.n_starts,self.maxiter,self.blas_threads)):
            raise ValueError('Candidate/start/iteration counts must be positive.')
        if not (0 < self.fraction_limit < 1):
            raise ValueError('fraction_limit must lie strictly between 0 and 1.')
        if self.mean_mode not in ('zero', 'profile'):
            raise ValueError('mean_mode must be zero or profile.')
        if self.bic_count not in ('scalar', 'locations', 'replicates'):
            raise ValueError('Unknown BIC count convention.')
        if len(self.marginal_sds) != 2 or min(self.marginal_sds) <= 0:
            raise ValueError('Two positive marginal standard deviations are required.')
        if not np.all(np.isfinite(self.marginal_sds)):
            raise ValueError('Marginal standard deviations must be finite.')
        if not np.isfinite(self.nugget_variance) or self.nugget_variance < 0:
            raise ValueError('nugget_variance must be finite and nonnegative.')
        if self.cross_nu_gap_max <= 0 or self.gaussian_gap_max <= 0:
            raise ValueError('Cross-gap upper bounds must be positive.')
        if not 0 < self.gaussian_odd_gap_min < self.gaussian_gap_max:
            raise ValueError('Gaussian odd gap bounds must be positive and ordered.')
        lo, hi = self.marginal_nu_bounds
        if not 0 < lo < hi:
            raise ValueError('Smoothness bounds must be positive and ordered.')
        allowed = {'nu11', 'nu22', 'nu_even', 'nu_odd'}
        if set(self.fixed_smoothness) - allowed:
            raise ValueError('Unknown fixed_smoothness parameter.')
        if any(not np.isfinite(v) or v <= 0 for v in self.fixed_smoothness.values()):
            raise ValueError('Fixed smoothness values must be positive and finite.')


class Geometry:
    """Cached exact distances/angles; C[i,j] uses h = location[j]-location[i]."""

    def __init__(self, locations, direction_degrees=0.0):
        loc = np.asarray(locations, dtype=float)
        if loc.ndim != 2 or loc.shape[1] != 2 or len(loc) < 2 or not np.isfinite(loc).all():
            raise ValueError('locations must be a finite (N,2) array with N>=2.')
        if len(np.unique(loc, axis=0)) != len(loc):
            raise ValueError('Duplicate sites are not supported; aggregate or model replicate errors explicitly.')
        if not np.isfinite(direction_degrees):
            raise ValueError('direction_degrees must be finite.')
        self.locations = loc.copy()
        self.n = len(loc)
        self.d = 2
        angle = np.deg2rad(direction_degrees)
        self.u0 = np.array([np.cos(angle), np.sin(angle)])
        lag = loc[None, :, :] - loc[:, None, :]
        self.r2 = np.einsum('ijk,ijk->ij', lag, lag)
        self.r = np.sqrt(self.r2)
        self.unique_r, inverse = np.unique(self.r, return_inverse=True)
        self.radial_index = inverse.reshape(self.r.shape)
        cosine = np.zeros_like(self.r)
        np.divide(lag @ self.u0, self.r, out=cosine, where=self.r > 0)
        self.cosine = np.clip(cosine, -1.0, 1.0)
        self.design = np.tile(np.eye(2), (self.n, 1))
        self._angular = {}
        self.direction_degrees = float(direction_degrees)

    def angular(self, p):
        if p not in self._angular:
            if p < 1 or p % 2 != 1:
                raise ValueError('p must be positive and odd.')
            a = eval_chebyt(p, self.cosine)
            # Enforce the exact known parity against floating-point evaluation noise.
            a = 0.5 * (a - a.T)
            np.fill_diagonal(a, 0.0)
            self._angular[p] = a
        return self._angular[p]

    def expand(self, values):
        return np.asarray(values)[self.radial_index]


def gaussian_radial(r, a, p=None):
    r = np.asarray(r, dtype=float)
    if not np.isfinite(a) or a <= 0 or np.any(r < 0) or not np.isfinite(r).all():
        raise ValueError('Gaussian radius/scale outside its domain.')
    if p is None:
        return np.exp(-(r / a)**2)
    if p < 1 or p % 2 != 1:
        raise ValueError('Odd degree required.')
    out = np.zeros_like(r)
    positive = r > 0
    logs = p * (math.log(2.0) - 2*math.log(a) + np.log(r[positive])) - (r[positive]/a)**2
    out[positive] = ((-1)**((p-1)//2)) * np.exp(logs)
    return out


def matern_radial(r, a, nu, p=None):
    """Normalized even or odd radial inverse; exact zero-lag assignments.

    log(kve(order,z))-z avoids large-z underflow in the intermediate K.
    A small-z leading expansion is used ONLY if the special function overflows
    and its mathematical remainder is negligible in that regime. Other numerical
    failures are rejected, never replaced with zero covariance.
    """
    r = np.asarray(r, dtype=float)
    if a <= 0 or nu <= 0 or not np.isfinite([a, nu]).all() or np.any(r < 0) or not np.isfinite(r).all():
        raise ValueError('Matern radius/scale/smoothness outside its domain.')
    if p is not None and (p < 1 or p % 2 != 1 or nu <= p/2):
        raise ValueError('Odd Matern requires positive odd p and nu>p/2.')
    out = np.zeros_like(r) if p is not None else np.ones_like(r)
    positive = r > 0
    z = a*r[positive]
    order = nu if p is None else abs(p-nu)
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        logk = np.log(kve(order, z)) - z
    bad = ~np.isfinite(logk)
    if bad.any():
        recoverable = bad & (z < 1e-8) & (order > 1e-4)
        logk[recoverable] = gammaln(order)+(order-1)*math.log(2.0)-order*np.log(z[recoverable])
        if not np.isfinite(logk).all():
            raise FloatingPointError('Bessel evaluation failed; tighten parameter bounds or inspect coordinate units.')
    logs = (1-nu)*math.log(2.0)-gammaln(nu)+nu*np.log(z)+logk
    values = np.exp(logs)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite normalized Matern covariance.')
    if p is not None:
        values *= (-1)**((p-1)//2)
    else:
        if np.any(values > 1.0+1e-9):
            raise FloatingPointError('Matern numerical evaluation exceeds its unit sill.')
        values = np.minimum(values, 1.0)
    out[positive] = values
    return out


def covariance_matrix(spec, params, geometry, sds=(1.0, 1.0), nugget_variance=0.0):
    """Assemble a 2N x 2N interleaved matrix; no eigendecomposition/projection."""
    radial = geometry.unique_r
    if spec.family == 'G':
        k11 = geometry.expand(gaussian_radial(radial, params['a11']))
        k22 = geometry.expand(gaussian_radial(radial, params['a22']))
    else:
        k11 = geometry.expand(matern_radial(radial, params['a11'], params['nu11']))
        k22 = geometry.expand(matern_radial(radial, params['a22'], params['nu22']))
    cross = np.zeros_like(k11)
    rho = params['rho']
    if spec.has_even and params['weight_even'] != 0 and rho != 0:
        e = (gaussian_radial(radial, params['a_even']) if spec.family == 'G'
             else matern_radial(radial, params['a_even'], params['nu_even']))
        cross += rho*params['weight_even']*geometry.expand(e)
    if spec.has_odd and params['weight_odd'] != 0 and rho != 0:
        o = (gaussian_radial(radial, params['a_odd'], spec.p) if spec.family == 'G'
             else matern_radial(radial, params['a_odd'], params['nu_odd'], spec.p))
        cross += rho*params['weight_odd']*geometry.expand(o)*geometry.angular(spec.p)
    c = np.empty((2*geometry.n, 2*geometry.n), dtype=float)
    c[0::2, 0::2] = sds[0]**2*k11
    c[1::2, 1::2] = sds[1]**2*k22
    c[0::2, 1::2] = sds[0]*sds[1]*cross
    c[1::2, 0::2] = sds[0]*sds[1]*cross.T
    if nugget_variance:
        c.flat[::c.shape[0]+1] += nugget_variance
    if not np.isfinite(c).all():
        raise FloatingPointError('Nonfinite covariance matrix.')
    return c


class ParameterSpace:
    """Named free parameters mapped to a spectrally admissible model.

    Pure components use continuous-frequency maxima. Mixed models use the
    sufficient sum of component maxima, which can be conservative. rho_fraction
    is bounded; the actual odd amplitude rho is NOT restricted to [-1,1].
    """
    def __init__(self, spec, geometry, config):
        config.validate()
        self.spec, self.geometry, self.config = spec, geometry, config
        positive = geometry.unique_r[geometry.unique_r > 0]
        lo, hi = config.length_bounds or (max(0.25*positive.min(), 1e-4), 3*positive.max())
        if not 0 < lo < hi or not np.isfinite([lo, hi]).all():
            raise ValueError('length_bounds must be finite, positive and increasing.')
        self.names, self.bounds = [], []
        self.add('log_length11', (np.log(lo), np.log(hi)))
        self.add('log_length22', (np.log(lo), np.log(hi)))
        if spec.family == 'G':
            if spec.has_even:
                self.add('gap_even', (0.0, config.gaussian_gap_max))
            if spec.has_odd:
                self.add('log_gap_odd', (np.log(config.gaussian_odd_gap_min), np.log(config.gaussian_gap_max)))
        else:
            if spec.has_even:
                self.add('log_length_even', (np.log(lo), np.log(hi)))
            if spec.has_odd:
                self.add('log_length_odd', (np.log(lo), np.log(hi)))
            for key in ('nu11', 'nu22'):
                if key not in config.fixed_smoothness:
                    self.add(key, config.marginal_nu_bounds)
            for part in ('even', 'odd'):
                active = spec.has_even if part == 'even' else spec.has_odd
                if active and 'nu_'+part not in config.fixed_smoothness:
                    self.add('nu_gap_'+part, (0.0, config.cross_nu_gap_max))
        if spec.kind == 'mixed':
            # Signed rho plus theta in [0,pi] covers the full coefficient plane.
            self.add('theta', (0.0, np.pi))
        self.add('rho_fraction', (-config.fraction_limit, config.fraction_limit))
        self.bounds = np.asarray(self.bounds, dtype=float)

    def add(self, name, bounds):
        self.names.append(name)
        self.bounds.append(bounds)

    def decode(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self.names),) or not np.isfinite(x).all():
            raise ValueError('Invalid optimizer parameter vector.')
        if np.any(x < self.bounds[:,0]) or np.any(x > self.bounds[:,1]):
            raise ValueError('Optimizer parameter vector is outside the configured bounds.')
        z = dict(zip(self.names, x))
        spec, cfg = self.spec, self.config
        theta = float(z.get('theta', 0.0 if spec.kind == 'even' else np.pi/2))
        # Treat the mathematical endpoints exactly, including floating pi/2.
        we = 0.0 if spec.kind == 'odd' or theta == np.pi/2 else float(np.cos(theta))
        wo = 0.0 if spec.kind == 'even' or theta in (0.0, np.pi) else float(np.sin(theta))
        a11, a22 = np.exp(z['log_length11']), np.exp(z['log_length22'])
        if spec.family == 'M':
            a11, a22 = 1/a11, 1/a22
        par = {'a11':float(a11), 'a22':float(a22)}
        if spec.family == 'G':
            base = 0.5*(a11*a11+a22*a22)
            if spec.has_even:
                par['a_even'] = float(np.nextafter(np.sqrt(base*(1+z['gap_even'])), np.inf))
            if spec.has_odd:
                par['a_odd'] = float(np.sqrt(base*(1+np.exp(z['log_gap_odd']))))
        else:
            for key in ('nu11', 'nu22'):
                par[key] = float(cfg.fixed_smoothness[key] if key in cfg.fixed_smoothness else z[key])
            avg = (par['nu11']+par['nu22'])/2
            for part in ('even', 'odd'):
                active = spec.has_even if part == 'even' else spec.has_odd
                if active:
                    par['a_'+part] = float(np.exp(-z['log_length_'+part]))
                    minimum = avg+(spec.p/2 if part == 'odd' else 0)
                    key = 'nu_'+part
                    par[key] = float(cfg.fixed_smoothness[key] if key in cfg.fixed_smoothness
                                     else np.nextafter(minimum+z['nu_gap_'+part], np.inf))
                    weight = we if part == 'even' else wo
                    if weight != 0 and par[key] < minimum:
                        raise ValueError('Fixed cross-smoothness violates the spectral tail condition.')
        logs = {}
        for part, active, weight in [('even', spec.has_even, we), ('odd', spec.has_odd, wo)]:
            if active and weight != 0:
                logs[part] = log_component_sup({'G':'gaussian','M':'matern'}[spec.family], a11, a22, par['a_'+part],
                    nu11=par.get('nu11'), nu22=par.get('nu22'), nuc=par.get('nu_'+part),
                    p=spec.p, odd=(part == 'odd'), d=2)
        if spec.kind == 'even':
            log_bound = logs['even']
        elif spec.kind == 'odd':
            log_bound = logs['odd']
        else:
            terms = [2*np.log(abs(w))+logs[k] for k,w in [('even',we),('odd',wo)] if w != 0]
            log_bound = float(np.logaddexp.reduce(terms))
        if not np.isfinite(log_bound):
            raise ValueError('No finite spectral amplitude bound for these parameters.')
        rho_max = float(np.exp(-0.5*log_bound))
        if not np.isfinite(rho_max) or rho_max <= 0:
            raise FloatingPointError('Amplitude bound cannot be represented at these parameter scales.')
        par.update(rho_fraction=float(z['rho_fraction']), rho_max=rho_max,
                   rho=float(z['rho_fraction']*rho_max), theta=theta,
                   weight_even=we, weight_odd=wo, log_spectral_bound=log_bound,
                   spectral_ratio_bound=float(z['rho_fraction']**2))
        if spec.family == 'M' and spec.has_odd and wo != 0:
            nu_star = par['nu_odd']-spec.p/2
            # b_2(nu)=Gamma(nu+1)/Gamma(nu)=nu.
            par['nu_star'] = nu_star
            par['rho_eff'] = par['rho']*par['nu_odd']/nu_star
        return par

    def vector_from_named(self, values):
        x = self.bounds.mean(axis=1)
        for i, key in enumerate(self.names):
            if key in values:
                x[i] = np.clip(values[key], *self.bounds[i])
        return x


class LikelihoodProblem:
    """One factorization and one batched triangular solve per candidate."""
    def __init__(self, space, observations):
        self.space = space
        y = np.asarray(observations, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[0] != 2*space.geometry.n or y.shape[1] < 1 or not np.isfinite(y).all():
            raise ValueError('observations must be finite and interleaved, shape (2*N,R).')
        self.y = np.asfortranarray(y)
        self.evaluations = 0
        self.failures = {}
        self.best = None
        self._cached = lru_cache(maxsize=512)(self._evaluate)

    def _evaluate(self, key):
        self.evaluations += 1
        try:
            cfg, geo = self.space.config, self.space.geometry
            par = self.space.decode(key)
            c = covariance_matrix(self.space.spec, par, geo, cfg.marginal_sds, cfg.nugget_variance)
            factor = cholesky(c, lower=True, check_finite=False, overwrite_a=True)
            if cfg.mean_mode == 'profile':
                rhs = np.column_stack([self.y, geo.design])
                white = solve_triangular(factor, rhs, lower=True, check_finite=False)
                wy, wx = white[:, :self.y.shape[1]], white[:, self.y.shape[1]:]
                beta = np.linalg.solve(wx.T@wx, wx.T@wy.mean(axis=1))
                resid = wy-wx@beta[:, None]
            else:
                beta = np.zeros(2)
                resid = solve_triangular(factor, self.y, lower=True, check_finite=False)
            m, reps = self.y.shape
            nll = 0.5*(np.sum(resid*resid)+reps*(2*np.log(np.diag(factor)).sum()+m*np.log(2*np.pi)))
            if not np.isfinite(nll):
                raise FloatingPointError('Nonfinite likelihood.')
            result = (float(nll), par, beta.tolist())
            if self.best is None or nll < self.best['nll']:
                self.best = dict(nll=float(nll), x=np.array(key), params=par, means=beta.tolist())
            return result
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
            reason = type(exc).__name__+': '+str(exc)
            self.failures[reason] = self.failures.get(reason, 0)+1
            return (np.inf, None, None)

    def evaluate(self, x):
        return self._cached(tuple(np.asarray(x, dtype=float)))

    def objective(self, x):
        # Finite penalty keeps finite differences defined at rejected candidates;
        # only genuinely finite likelihoods may become selected results.
        val = self.evaluate(x)[0]
        return val if np.isfinite(val) else 1e50



































































# json_serialization.py
"""Portable serialization shared by the standalone experiment components."""
import json
from pathlib import Path
import numpy as np

def _json_safe(value):
    if isinstance(value,dict):
        return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value,(np.integer,)):
        return int(value)
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,Path):
        return str(value)
    return value


# robust_fitting.py
"""Bounded MLE with scaled coordinates and independent local checks.

The caller makes ``mle_core`` importable.  No generating parameters are used
unless the caller explicitly supplies them in ``warm_starts``; such starts
should only be used for a separately reported diagnostic.  A successful local
check is evidence about local numerical stationarity, never global optimality.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import time
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc



@dataclass
class RobustOptions:
    gradient_step: float = 2e-5
    gradient_tolerance: float = 5e-4
    directional_tolerance: float = 2e-7
    polish_rounds: int = 2
    fallback_maxiter: int = 500
    check_random_directions: int = 8


class ScaledObjective:
    """Unit-box coordinates; finite differences adapt at invalid candidates."""
    def __init__(self, problem, options):
        self.problem, self.options = problem, options
        self.lower = problem.space.bounds[:, 0]
        self.width = np.ptp(problem.space.bounds, axis=1)
        self.scale = float(problem.y.size)
        self.bad_gradient_coordinates = 0

    def to_unit(self, x):
        return np.clip((np.asarray(x)-self.lower)/self.width, 0, 1)

    def to_native(self, z):
        # Clipping here only protects floating-point bound arithmetic.  It never
        # modifies a covariance or its eigenvalues.
        return np.clip(self.lower+self.width*np.asarray(z),
                       self.problem.space.bounds[:, 0], self.problem.space.bounds[:, 1])

    def finite(self, z):
        return self.problem.evaluate(self.to_native(z))[0]/self.scale

    def __call__(self, z):
        value = self.finite(z)
        return float(value) if np.isfinite(value) else 1e6

    def gradient(self, z):
        z = np.asarray(z, dtype=float)
        f0 = self.finite(z)
        if not np.isfinite(f0):
            # An invalid line-search point must not become the retained fit.
            self.bad_gradient_coordinates += len(z)
            return np.zeros(len(z))
        grad = np.zeros(len(z))
        for j in range(len(z)):
            samples = {}
            for sign in (-1, 1):
                h = min(self.options.gradient_step, z[j] if sign < 0 else 1-z[j])
                for _ in range(9):
                    if h < 1e-10:
                        break
                    trial = z.copy()
                    trial[j] += sign*h
                    value = self.finite(trial)
                    if np.isfinite(value):
                        samples[sign] = (h, value)
                        break
                    h *= 0.25
            if -1 in samples and 1 in samples:
                hm, fm = samples[-1]
                hp, fp = samples[1]
                # Unequal-step second-order three-point derivative.
                grad[j] = (hm*hm*(fp-f0)+hp*hp*(f0-fm))/(hp*hm*(hp+hm))
            elif 1 in samples:
                h, fp = samples[1]
                grad[j] = (fp-f0)/h
            elif -1 in samples:
                h, fm = samples[-1]
                grad[j] = (f0-fm)/h
            else:
                self.bad_gradient_coordinates += 1
                grad[j] = np.nan
        # NaN gradients abort the numerical path rather than silently certifying
        # it.  The final independent check explicitly requires finite gradients.
        return grad


def local_stationarity(objective, z, seed=1729):
    """Projected gradient plus feasible finite-scale coordinate/random probes."""
    z = np.asarray(z).copy()
    f0 = objective.finite(z)
    grad = objective.gradient(z)
    projected = grad.copy()
    projected[(z <= 1e-7) & (grad >= 0)] = 0
    projected[(z >= 1-1e-7) & (grad <= 0)] = 0
    directions = list(np.eye(len(z))) + list(-np.eye(len(z)))
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(objective.options.check_random_directions, len(z)))
    if len(random):
        random /= np.linalg.norm(random, axis=1)[:, None]
        directions += list(random) + list(-random)
    best, best_z, finite_count, rejected = f0, z.copy(), 0, 0
    for radius in (1e-3, 2e-4):
        for direction in directions:
            trial = np.clip(z+radius*direction, 0, 1)
            if np.array_equal(trial, z):
                continue
            value = objective.finite(trial)
            if np.isfinite(value):
                finite_count += 1
                if value < best:
                    best, best_z = value, trial
            else:
                rejected += 1
    norm = float(np.max(np.abs(projected))) if np.isfinite(projected).all() else float('inf')
    improvement = max(0., f0-best)
    passed = (norm <= objective.options.gradient_tolerance and
              improvement <= objective.options.directional_tolerance)
    diagnostic = dict(passed=bool(passed), projected_gradient_inf=norm,
        gradient_tolerance=objective.options.gradient_tolerance,
        best_probe_improvement_per_scalar=improvement,
        best_probe_improvement_nll=improvement*objective.scale,
        directional_tolerance_per_scalar=objective.options.directional_tolerance,
        finite_probes=finite_count, rejected_probes=rejected,
        free_coordinates_at_box_boundary=int(np.sum((z <= 1e-7) | (z >= 1-1e-7))),
        gradient=grad.tolist(), checked_unit_coordinates=z.tolist())
    return diagnostic, best_z


def reflected_mixed_start(space, x):
    """Explore the other odd-sign branch using fitted parameters only.

    The reflection preserves the even spectral coefficient and reverses the odd
    coefficient. At theta=0/pi the reflected point is exactly the same model,
    but the other chart opens the opposite feasible odd direction. Move this
    seed slightly inside the chart and reinitialize inactive odd shapes to
    generic values so a previously irrelevant shape cannot obstruct that move.
    """
    values = dict(zip(space.names, np.asarray(x, dtype=float)))
    old_theta = values['theta']
    values['theta'] = np.pi-old_theta
    values['rho_fraction'] = -values['rho_fraction']
    if min(old_theta, np.pi-old_theta) <= 1e-6:
        values['theta'] = 0.15 if values['theta'] < np.pi/2 else np.pi-0.15
        if space.spec.family == 'G':
            values['log_gap_odd'] = np.log(0.5)
        else:
            values['log_length_odd'] = (values['log_length11']+values['log_length22'])/2
            if 'nu_gap_odd' in values:
                values['nu_gap_odd'] = 0.5
    return space.vector_from_named(values)


def robust_fit_model(spec, geometry, observations, config=None, warm_starts=None,
                     optimizer_options=None):
    """Return a fit_model-compatible result, with honestly labelled local checks.

    ``config.n_candidates/n_starts/maxiter`` control the primary search budget.
    Random candidate starts and generic range/correlation starts depend only on
    the observations' geometry, configuration, and random seed.
    """
    config = config or FitConfig()
    options = optimizer_options or RobustOptions()
    started = time.perf_counter()
    space = ParameterSpace(spec, geometry, config)
    problem = LikelihoodProblem(space, observations)
    objective = ScaledObjective(problem, options)
    threads = nullcontext()
    try:
        from threadpoolctl import threadpool_limits
        threads = threadpool_limits(limits=config.blas_threads, user_api='blas')
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        warnings.warn('BLAS thread control unavailable: '+str(exc))
    runs, diagnostic = [], None
    with threads:
        sample = qmc.LatinHypercube(d=len(space.names), seed=config.seed).random(config.n_candidates)
        candidates = [objective.to_native(row) for row in sample]
        # Cross ranges start near the marginal ranges, not halfway through a
        # potentially enormous gap interval. These are generic, not truth starts.
        for length_fraction, rho in ((0.25, 0.35), (0.50, -0.35), (0.70, 0.65), (0.45, 0.0)):
            values = dict(rho_fraction=rho, theta=np.pi/4, gap_even=0.3,
                log_gap_odd=np.log(0.5), nu11=1.5, nu22=1.5,
                nu_gap_even=0.5, nu_gap_odd=0.5)
            for name, bounds in zip(space.names, space.bounds):
                if name.startswith('log_length'):
                    values[name] = bounds[0]+length_fraction*(bounds[1]-bounds[0])
            candidates.append(space.vector_from_named(values))
        candidates += [space.vector_from_named(w) for w in (warm_starts or [])]
        scored = [(problem.evaluate(x)[0], x) for x in candidates]
        scored = sorted((pair for pair in scored if np.isfinite(pair[0])), key=lambda pair:pair[0])

        def run_local(z, label, method='L-BFGS-B'):
            before = problem.evaluations
            if method == 'L-BFGS-B':
                result = minimize(objective, z, jac=objective.gradient, method=method,
                    bounds=[(0, 1)]*len(z), options=dict(maxiter=config.maxiter,
                    ftol=min(config.ftol, 1e-11), gtol=1e-5, maxls=35, maxcor=12))
            else:
                # A finite-scale simplex can leave a spuriously stopped line
                # search.  It is never used as the sole convergence certificate.
                simplex = np.tile(z, (len(z)+1, 1))
                for j in range(len(z)):
                    simplex[j+1,j] += 0.025 if z[j] <= 0.975 else -0.025
                result = minimize(objective, z, method='Nelder-Mead',
                    bounds=[(0,1)]*len(z), options=dict(maxiter=options.fallback_maxiter,
                    xatol=1e-5, fatol=1e-9, adaptive=True, initial_simplex=simplex))
            nll = problem.evaluate(objective.to_native(result.x))[0]
            runs.append(dict(stage=label, method=method, success=bool(result.success),
                message=str(result.message), nit=int(result.nit), nfev=int(result.nfev),
                actual_covariance_evaluations=problem.evaluations-before,
                nll=float(nll), x=objective.to_native(result.x).tolist()))

        for idx, (_, candidate) in enumerate(scored[:config.n_starts]):
            run_local(objective.to_unit(candidate), 'multistart_'+str(idx))
        if spec.kind == 'mixed' and problem.best is not None:
            reflected = reflected_mixed_start(space, problem.best['x'])
            if np.isfinite(problem.evaluate(reflected)[0]):
                run_local(objective.to_unit(reflected), 'reflected_mixed_phase')
        if problem.best is not None:
            for round_index in range(options.polish_rounds+1):
                checked = objective.to_unit(problem.best['x'])
                diagnostic, _ = local_stationarity(objective, checked, config.seed+991)
                if diagnostic['passed']:
                    break
                if round_index == options.polish_rounds:
                    break
                run_local(objective.to_unit(problem.best['x']), 'polish_'+str(round_index))
                recheck, _ = local_stationarity(objective,
                    objective.to_unit(problem.best['x']), config.seed+991)
                if not recheck['passed']:
                    run_local(objective.to_unit(problem.best['x']),
                              'simplex_escape_'+str(round_index), method='Nelder-Mead')
            # Preserve every better finite value, and check that exact selected
            # point without repeatedly chasing sub-tolerance probe differences.
            selected = problem.best.copy()
            diagnostic, _ = local_stationarity(objective,
                objective.to_unit(selected['x']), config.seed+991)
        else:
            selected = None
    reps = problem.y.shape[1]
    k = len(space.names)+(2 if config.mean_mode == 'profile' else 0)
    bic_n = {'scalar':problem.y.size, 'locations':geometry.n*reps, 'replicates':reps}[config.bic_count]
    result = dict(model=spec.name, p=spec.p if spec.has_odd else None, k=k,
        N=geometry.n, R=reps, bic_count=config.bic_count, bic_n=bic_n,
        seconds=time.perf_counter()-started, evaluations=problem.evaluations,
        optimizer_runs=runs, evaluation_failures=problem.failures,
        local_diagnostics=diagnostic,
        best_evaluated_nll=(None if problem.best is None else problem.best['nll']),
        validity_method='continuous-frequency component maxima; sufficient mixed bound',
        mean_mode=config.mean_mode, nugget_variance=config.nugget_variance,
        marginal_sds=list(config.marginal_sds), seed=config.seed,
        optimizer='unit-box multistart with adaptive finite differences and local probes')
    if selected is None:
        result.update(status='failed', converged=False, nll=None, aic=None, bic=None,
            params={}, free_parameters={}, fitted_means=None, bounds_reached=[])
        return result
    x = selected['x']
    distances = np.minimum(x-space.bounds[:,0], space.bounds[:,1]-x)
    reached = [name for name,gap,width in zip(space.names, distances,np.ptp(space.bounds,axis=1))
               if gap <= 1e-4*max(width,1.0)]
    passed = diagnostic['passed']
    result.update(status='locally_checked' if passed else 'best_finite_unconverged',
        converged=bool(passed), nll=selected['nll'], aic=2*selected['nll']+2*k,
        bic=2*selected['nll']+k*np.log(bic_n), params=selected['params'],
        free_parameters=dict(zip(space.names,x.tolist())), fitted_means=selected['means'],
        bounds_reached=reached)
    return result


# gaussian_refinement.py
"""Prespecified second-stage Gaussian searches, independent of generating truth.

Apply to every Gaussian dataset. All diversification starts are optimized;
initial likelihood ranking never drops an alternative sign/mixing branch.
"""
from contextlib import nullcontext
from copy import deepcopy
import time
import warnings
import numpy as np
from scipy.optimize import minimize




def gaussian_diversification_starts(space, baseline_free_parameters):
    """Generic cross gaps and sign/quadrant starts, retaining fitted marginals."""
    if space.spec.family != 'G' or space.spec.kind not in ('odd', 'mixed'):
        raise ValueError('Gaussian refinement supports G_odd and G_mixed.')
    starts = []
    angles = (None,) if space.spec.kind == 'odd' else (np.pi/4, 3*np.pi/4)
    for angle in angles:
        for fraction in (-0.5, 0.5):
            values = dict(baseline_free_parameters)
            values['log_gap_odd'] = np.log(0.5)
            values['rho_fraction'] = fraction
            if angle is not None:
                values['gap_even'] = 0.5
                values['theta'] = angle
            starts.append(space.vector_from_named(values))
    return starts


def refine_gaussian_fit(spec, geometry, observations, baseline_result, config):
    """Return best checked candidate; preserve the entire original fit for audit.

    Only ``free_parameters`` from ``baseline_result`` initialize optimization.
    Metadata such as its likelihood at the generating truth are never read by
    the search, stopping criteria, or selection rule.
    """
    started = time.perf_counter()
    if spec.family != 'G' or spec.kind not in ('odd', 'mixed'):
        raise ValueError('This refinement applies only to Gaussian odd/mixed fits.')
    original = deepcopy(baseline_result)
    result = deepcopy(baseline_result)
    result['initial_fit'] = original
    space = ParameterSpace(spec, geometry, config)
    problem = LikelihoodProblem(space, observations)
    options = RobustOptions(polish_rounds=1, fallback_maxiter=350)
    objective = ScaledObjective(problem, options)
    free = baseline_result.get('free_parameters')
    if not free:
        result['refinement'] = dict(status='no_finite_baseline', seconds=0.,
            algorithm='all prescribed sign/quadrant starts; generic cross gaps')
        return result
    base_x = space.vector_from_named(free)
    seed_vectors = gaussian_diversification_starts(space, free)
    runs = []
    threads = nullcontext()
    try:
        from threadpoolctl import threadpool_limits
        threads = threadpool_limits(limits=config.blas_threads, user_api='blas')
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        warnings.warn('BLAS thread control unavailable: '+str(exc))
    with threads:
        initial_nll = problem.evaluate(base_x)[0]

        def run(z, label, method='L-BFGS-B'):
            count = problem.evaluations
            if method == 'L-BFGS-B':
                solved = minimize(objective,z,jac=objective.gradient,method=method,
                    bounds=[(0.,1.)]*len(z), options=dict(maxiter=config.maxiter,
                    ftol=min(config.ftol,1e-11),gtol=1e-5,maxls=35,maxcor=12))
            else:
                simplex = np.tile(z,(len(z)+1,1))
                for j in range(len(z)):
                    simplex[j+1,j] += .025 if z[j] <= .975 else -.025
                solved = minimize(objective,z,method=method,bounds=[(0.,1.)]*len(z),
                    options=dict(maxiter=350,xatol=1e-5,fatol=1e-9,adaptive=True,
                                 initial_simplex=simplex))
            value = problem.evaluate(objective.to_native(solved.x))[0]
            runs.append(dict(stage=label,method=method,success=bool(solved.success),
                message=str(solved.message),nit=int(solved.nit),nfev=int(solved.nfev),
                actual_covariance_evaluations=problem.evaluations-count,
                nll=float(value),x=objective.to_native(solved.x).tolist()))
            return solved

        for index, vector in enumerate(seed_vectors):
            # Never rank or discard these prescribed alternatives by their
            # initial likelihood: every finite start enters local optimization.
            if not np.isfinite(problem.evaluate(vector)[0]):
                runs.append(dict(stage='prescribed_'+str(index),method='none',
                    success=False,message='Prescribed start has invalid numerical likelihood',
                    nit=0,nfev=0,nll=float('inf'),x=vector.tolist()))
                continue
            solved = run(objective.to_unit(vector),'prescribed_'+str(index))
            if solved.nit <= 2:
                gradient = objective.gradient(solved.x)
                projected = gradient.copy()
                projected[(solved.x <= 1e-7) & (gradient >= 0)] = 0
                projected[(solved.x >= 1-1e-7) & (gradient <= 0)] = 0
                if not np.isfinite(projected).all() or np.max(np.abs(projected)) > options.gradient_tolerance:
                    escape = run(solved.x,'prescribed_escape_'+str(index),'Nelder-Mead')
                    run(escape.x,'prescribed_repolish_'+str(index))
        if problem.best is None:
            result['refinement'] = dict(status='no_finite_candidate',
                seconds=time.perf_counter()-started,optimizer_runs=runs)
            return result
        diagnostic,_ = local_stationarity(objective,
            objective.to_unit(problem.best['x']),config.seed+12347)
        if not diagnostic['passed']:
            run(objective.to_unit(problem.best['x']),'final_polish')
        selected = problem.best.copy()
        diagnostic,_ = local_stationarity(objective,
            objective.to_unit(selected['x']),config.seed+12347)
    elapsed = time.perf_counter()-started
    x = selected['x']
    distance = np.minimum(x-space.bounds[:,0],space.bounds[:,1]-x)
    reached = [name for name,gap,width in zip(space.names,distance,np.ptp(space.bounds,axis=1))
               if gap <= 1e-4*max(width,1.)]
    k = len(space.names)+(2 if config.mean_mode == 'profile' else 0)
    reps = problem.y.shape[1]
    bic_n = {'scalar':problem.y.size,'locations':geometry.n*reps,'replicates':reps}[config.bic_count]
    passed = diagnostic['passed']
    result.update(status='locally_checked' if passed else 'best_finite_unconverged',
        converged=bool(passed),locally_validated=bool(passed),local_diagnostics=diagnostic,
        nll=selected['nll'],aic=2*selected['nll']+2*k,bic=2*selected['nll']+k*np.log(bic_n),
        params=selected['params'],free_parameters=dict(zip(space.names,x.tolist())),
        fitted_means=selected['means'],bounds_reached=reached,
        best_evaluated_nll=problem.best['nll'],
        seconds=float(baseline_result.get('seconds',0.))+elapsed,
        evaluations=int(baseline_result.get('evaluations',0))+problem.evaluations,
        optimizer_runs=list(baseline_result.get('optimizer_runs',[]))+runs)
    failures = dict(baseline_result.get('evaluation_failures',{}))
    for key,count in problem.failures.items():
        failures[key] = failures.get(key,0)+count
    result['evaluation_failures'] = failures
    result['refinement'] = dict(status='completed',
        algorithm='all prescribed sign/quadrant starts; generic cross gaps 0.5; fitted marginal scales',
        prescribed_starts=len(seed_vectors),initial_nll=float(initial_nll),
        selected_nll=selected['nll'],nll_improvement=float(initial_nll-selected['nll']),
        seconds=elapsed,evaluations=problem.evaluations,optimizer_runs=runs,
        local_diagnostics=diagnostic)
    # Truth diagnostics are intentionally left untouched here. The caller can
    # recompute them afterwards as reporting metadata, outside the estimator.
    return result


# figure_fields.py
"""Corrected angular and joint-field experiments for manuscript Figures 1--3.

The caller makes the corrected ``mle_core`` module available on sys.path.
All simulated covariances use h = location[j] - location[i], matching that core.
"""
from pathlib import Path
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular





def _ff_save(fig, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in ('png', 'pdf'):
        path = folder / (stem + '.' + extension)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        paths[extension] = str(path)
    plt.close(fig)
    return paths


def _ff_rho_bounds(family, params, p, angles):
    optional = {} if family == 'G' else dict(nu11=params['nu11'], nu22=params['nu22'])
    logs = []
    for part in ('even', 'odd'):
        kw = dict(optional)
        if family == 'M':
            kw['nuc'] = params['nu_' + part]
        logs.append(log_component_sup(
            'gaussian' if family == 'G' else 'matern', params['a11'], params['a22'],
            params['a_' + part], p=p, odd=part == 'odd', **kw))
    bounds = []
    for theta in angles:
        we = 0.0 if theta == np.pi/2 else np.cos(theta)
        wo = 0.0 if theta == 0 else np.sin(theta)
        limit = 1/np.sqrt(we*we*np.exp(logs[0]) + wo*wo*np.exp(logs[1]))
        if not abs(params['rho']) < limit:
            raise ValueError('A Figure 2/3 configuration violates the sufficient spectral bound.')
        bounds.append(float(limit))
    return bounds


def _ff_angular(folder, raw_folder):
    phi = np.linspace(0, 2*np.pi, 1201)
    degrees = (1, 3, 5)
    values = np.empty((2, 3, len(phi)))
    bounds = {'G': [], 'M': []}
    for j, p in enumerate(degrees):
        values[0, j] = .004*gaussian_radial(np.array([1.]), 1., p)[0]*np.cos(p*phi)
        values[1, j] = .7*matern_radial(np.array([1.]), 1., 4., p)[0]*np.cos(p*phi)
        bounds['G'].append(float(np.exp(-.5*log_component_sup(
            'gaussian', .5, .5, 1., p=p, odd=True))))
        bounds['M'].append(float(np.exp(-.5*log_component_sup(
            'matern', 1., 1., 1., nu11=1., nu22=1., nuc=4., p=p, odd=True))))
    if min(bounds['G']) <= .004 or min(bounds['M']) <= .7:
        raise ValueError('Angular configurations must share a valid amplitude over all p.')
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.1), sharex=True)
    for i, family in enumerate(('Gaussian', "Matérn")):
        for j, p in enumerate(degrees):
            ax = axes[i, j]
            ax.plot(phi, values[i, j], color=('#24618c' if i == 0 else '#a95325'), lw=1.9)
            ax.axhline(0, color='.65', lw=.7)
            ax.set_title(r'$p=%d$' % p)
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(-1.12*np.max(np.abs(values[i, j])), 1.12*np.max(np.abs(values[i, j])))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.set_xticks(np.arange(5)*np.pi/2)
            ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
            ax.grid(alpha=.18)
            if j == 0:
                ax.set_ylabel(family + r' $C_{12}(\mathbf{h})$')
            if i == 1:
                ax.set_xlabel(r'Angle $\phi$ from $\mathbf{u}_0=(1,0)$')
    fig.suptitle('Effect of odd degree at lag norm 1; vertical scales vary by panel', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    paths = _ff_save(fig, folder, 'figure_1_effect_of_p')
    raw_path = raw_folder / 'figure_1_angular_values.npz'
    np.savez_compressed(raw_path, phi=phi, p=np.asarray(degrees), gaussian=values[0], matern=values[1])
    return dict(paths=paths, raw_data=str(raw_path), rho_max_by_p=bounds,
                parameters=dict(G=dict(a11=.5, a22=.5, a_odd=1., rho=.004),
                                M=dict(a11=1., a22=1., a_odd=1., nu11=1., nu22=1., nu_odd=4., rho=.7),
                                p=list(degrees), lag_norm=1., u0=[1., 0.], marginal_sds=[1., 1.]),
                caption=('Effect of p=1,3,5 on the corrected odd cross-covariances at lag norm 1 '
                         'and direction (1,0). Gaussian parameters: a11=a22=0.5, a_odd=1, rho=0.004. '
                         'Matérn parameters: a11=a22=a_odd=1, nu11=nu22=1, nu_odd=4, rho=0.7. '
                         'Marginal variances equal one. Each family holds all parameters except p fixed; '
                         'all six full bivariate models pass their spectral validity bounds. '
                         'The panels show actual covariance amplitudes, with different vertical scales.'))


def _ff_gaussian_fft(params, angles, seed, grid_side=25, fft_side=128):
    """PSD Fourier quadrature, with error checked on every displayed pair lag.

    The FFT convention for an output X(s)=sum A(u) exp(i u.s) gives
    Cov(X1(s),X2(t))=sum E[A1 conj(A2)] exp(-i u.(t-s)).
    Therefore E[A1 conj(A2)] must be c+i*q when the manuscript has f12=c-i*q.
    The lower triangular spectral factor below uses A2=(c-i*q)/sqrt(f11)*W1+... .
    Real white-noise FFTs enforce conjugacy, including the DC/Nyquist modes.
    """
    spacing = 10./(grid_side-1)
    wave = 2*np.pi*np.fft.fftfreq(fft_side, d=spacing)
    ux, uy = np.meshgrid(wave, wave)
    radius2 = ux*ux + uy*uy
    du = 2*np.pi/(fft_side*spacing)
    density = lambda a: a*a/(4*np.pi)*np.exp(-a*a*radius2/4)
    f11, f22 = density(params['a11']), density(params['a22'])
    fe, qo = density(params['a_even']), ux*density(params['a_odd'])
    neg = (-np.arange(fft_side)) % fft_side
    # Oddness at periodic Nyquist modes is imposed before factorisation. The
    # omitted contribution is included in the deterministic covariance error.
    qo = .5*(qo-qo[np.ix_(neg, neg)])
    rng = np.random.default_rng(seed)
    w1 = np.fft.fft2(rng.standard_normal((fft_side, fft_side)))/fft_side
    w2 = np.fft.fft2(rng.standard_normal((fft_side, fft_side)))/fft_side
    l11 = np.sqrt(f11)
    fields = np.empty((len(angles), 2, grid_side, grid_side))
    errors = []
    modes = np.arange(-(grid_side-1), grid_side)
    hx, hy = np.meshgrid(modes*spacing, modes*spacing)
    r = np.hypot(hx, hy)
    cosine = np.divide(hx, r, out=np.zeros_like(r), where=r > 0)
    indices = np.mod(modes, fft_side)
    scale = fft_side**2*du*du
    marginal_errors = []
    for f, a in ((f11, params['a11']), (f22, params['a22'])):
        periodic = (np.fft.ifft2(f)*scale).real[np.ix_(indices, indices)]
        marginal_errors.append(float(np.max(np.abs(periodic-gaussian_radial(r, a)))))
    imaginary_max = 0.
    for k, theta in enumerate(angles):
        we = 0.0 if theta == np.pi/2 else np.cos(theta)
        wo = 0.0 if theta == 0 else np.sin(theta)
        f12 = params['rho']*(we*fe-1j*wo*qo)
        coherence2 = np.abs(f12)**2/(f11*f22)
        if not np.isfinite(coherence2).all() or coherence2.max() >= 1:
            raise ValueError('Gaussian Fourier covariance is not strictly positive definite.')
        l21 = f12/l11
        l22 = np.sqrt(f22*(1-coherence2))
        for component, amplitudes in enumerate((l11*w1, l21*w1+l22*w2)):
            complex_field = np.fft.ifft2(amplitudes)*(fft_side**2*du)
            imaginary_max = max(imaginary_max, float(np.max(np.abs(complex_field.imag))))
            fields[k, component] = complex_field.real[:grid_side, :grid_side]
        # This compares the covariance implied by the actual spectral factor,
        # not an empirical covariance from one noisy realization.
        factored_f12 = l11*l21
        numeric = (np.fft.ifft2(factored_f12)*scale).real[np.ix_(indices, indices)]
        exact = params['rho']*(we*gaussian_radial(r, params['a_even']) +
                              wo*gaussian_radial(r, params['a_odd'], 1)*cosine)
        error = float(np.max(np.abs(numeric-exact)))
        errors.append(dict(theta=float(theta), cross_covariance_max_abs_error=error,
                           spectral_squared_coherence_max=float(coherence2.max())))
    max_error = max(marginal_errors + [e['cross_covariance_max_abs_error'] for e in errors])
    if max_error > 1e-10 or imaginary_max > 1e-10:
        raise ArithmeticError('Fourier quadrature failed its deterministic accuracy check.')
    if not all(np.array_equal(fields[0, 0], fields[k, 0]) for k in range(len(angles))):
        raise AssertionError('The first field must stay identical across the angle sweep.')
    return fields, dict(method='periodic Fourier spectral quadrature', fft_side=fft_side,
                        grid_side=grid_side, spacing=spacing, period=fft_side*spacing,
                        frequency_spacing=du, maximum_abs_frequency=float(np.max(np.abs(wave))),
                        checked_displacement_count=int(r.size),
                        maximum_absolute_covariance_error_all_displayed_pair_lags=max_error,
                        marginal_max_abs_errors=marginal_errors, angle_checks=errors,
                        maximum_imaginary_roundoff=imaginary_max,
                        eigenvalue_clipping=False, nugget_variance=0.,
                        error_definition=('Maximum entrywise difference between the covariance induced by the '
                                          'actual finite spectral factor and the target continuous Gaussian '
                                          'covariance, at every lag occurring on the displayed grid; this includes '
                                          'periodization, frequency truncation, and Nyquist treatment.'))


def _ff_matern_cholesky(params, angles, seed, grid_side=22):
    side = np.linspace(0, 10, grid_side)
    xx, yy = np.meshgrid(side, side)
    geo = Geometry(np.column_stack((xx.ravel(), yy.ravel())), direction_degrees=0.)
    k11 = geo.expand(matern_radial(geo.unique_r, params['a11'], params['nu11']))
    k22 = geo.expand(matern_radial(geo.unique_r, params['a22'], params['nu22']))
    l11 = cholesky(k11, lower=True, check_finite=True)
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal((2, len(side)**2))
    first = l11 @ z1
    fields = np.empty((len(angles), 2, grid_side, grid_side))
    checks = []
    for k, theta in enumerate(angles):
        par = dict(params, theta=float(theta), weight_even=0. if theta == np.pi/2 else float(np.cos(theta)),
                   weight_odd=0. if theta == 0 else float(np.sin(theta)))
        full = covariance_matrix(ModelSpec('M', 'mixed', p=1), par, geo)
        cross = full[0::2, 1::2]
        q = solve_triangular(l11, cross, lower=True, check_finite=False)
        conditional = k22-q.T @ q
        l22 = cholesky(conditional, lower=True, check_finite=True)
        fields[k, 0] = first.reshape(grid_side, grid_side)
        fields[k, 1] = (q.T @ z1 + l22 @ z2).reshape(grid_side, grid_side)
        error = max(float(np.max(np.abs(l11 @ l11.T-k11))),
                    float(np.max(np.abs(l11 @ q-cross))),
                    float(np.max(np.abs(q.T @ q+l22 @ l22.T-k22))))
        checks.append(dict(theta=float(theta), covariance_factor_max_abs_error=error,
                           conditional_cholesky_min_diagonal=float(l22.diagonal().min())))
    if max(e['covariance_factor_max_abs_error'] for e in checks) > 1e-10:
        raise ArithmeticError('Conditional factorization does not reproduce the target covariance.')
    if not all(np.array_equal(fields[0, 0], fields[k, 0]) for k in range(len(angles))):
        raise AssertionError('The first field must stay identical across the angle sweep.')
    return fields, dict(method='direct conditional Cholesky, exact covariance up to floating-point error',
                        grid_side=grid_side, spacing=10./(grid_side-1), angle_checks=checks,
                        eigenvalue_clipping=False, nugget_variance=0.)


def _ff_field_plot(fields, angles, family, rho, folder, stem):
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 7.2), sharex=True, sharey=True)
    labels = [r'$\theta=0$ (even)', r'$\theta=\pi/4$ (mixed)', r'$\theta=\pi/2$ (odd)']
    limit = float(np.ceil(np.abs(fields).max()*2)/2)
    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            spacing = 10./(fields.shape[-1]-1)
            im = ax.imshow(fields[j, i], origin='lower',
                           extent=(-spacing/2, 10+spacing/2, -spacing/2, 10+spacing/2),
                           cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='bicubic')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            if i == 0:
                ax.set_title(labels[j], fontsize=10)
            if j == 0:
                ax.set_ylabel('Field %d\ny' % (i+1))
            if i == 1:
                ax.set_xlabel('x')
            ax.set_xticks([0, 5, 10])
            ax.set_yticks([0, 5, 10])
    fig.suptitle(family + r' bivariate fields: $p=1$, $\rho=%g$' % rho, fontsize=13)
    fig.subplots_adjust(left=.08, right=.89, bottom=.09, top=.90, wspace=.14, hspace=.13)
    cax = fig.add_axes([.92, .17, .018, .63])
    fig.colorbar(im, cax=cax, label='Field value (unit marginal variance)')
    return _ff_save(fig, folder, stem), limit


def generate_figures_1_to_3(outdir, seed=20260912):
    """Write PNG/PDF figures, raw arrays, and deterministic validation metadata."""
    outdir = Path(outdir)
    folder, raw_folder = outdir/'figures', outdir/'raw_data'
    folder.mkdir(parents=True, exist_ok=True)
    raw_folder.mkdir(parents=True, exist_ok=True)
    angles = np.array([0., np.pi/4, np.pi/2])
    metadata = {'figure_1': _ff_angular(folder, raw_folder)}
    configurations = [
        ('G', dict(a11=3., a22=3., a_even=4., a_odd=4., rho=.5), _ff_gaussian_fft, 'figure_2_gaussian_fields'),
        ('M', dict(a11=3., a22=3., a_even=4., a_odd=4., nu11=1.5, nu22=2.5,
                   nu_even=3.5, nu_odd=4.5, rho=.8), _ff_matern_cholesky, 'figure_3_matern_fields')]
    for index, (family, params, simulator, stem) in enumerate(configurations, start=2):
        family_seed = int(seed)+index
        rho_bounds = _ff_rho_bounds(family, params, 1, angles)
        fields, validation = simulator(params, angles, family_seed)
        paths, color_limit = _ff_field_plot(fields, angles, 'Gaussian' if family == 'G' else 'Matérn',
                                           params['rho'], folder, stem)
        raw_path = raw_folder/(stem+'.npz')
        np.savez_compressed(raw_path, fields=fields, theta=angles,
                            x=np.linspace(0, 10, fields.shape[-1]), y=np.linspace(0, 10, fields.shape[-1]),
                            seed=np.array(family_seed))
        common = ('Rows show Fields 1 and 2; columns show theta=0, pi/4, pi/2. '
                  'The same latent normal innovations are reused across columns, so Field 1 is identical; '
                  'only cross-dependence changes. All panels share a symmetric color scale. '
                  'Bicubic interpolation is a display operation only; the stored raw fields are unfiltered. '
                  'No eigenvalue clipping or nugget is used. Marginal variances are one, p=1, and u0=(1,0). ')
        if family == 'G':
            caption = ('Gaussian bivariate fields on a 25 by 25 grid over [0,10]^2, with '
                       'a11=a22=3, a_even=a_odd=4, rho=0.5 (changed from 0.85 to satisfy spectral validity '
                       'at every angle). Simulation uses a 128 by 128 periodic Fourier quadrature. '
                       'The maximum absolute covariance error over all displayed pair lags is %.3g. ' %
                       validation['maximum_absolute_covariance_error_all_displayed_pair_lags']) + common
        else:
            caption = ('Matérn bivariate fields on a 22 by 22 grid over [0,10]^2, with '
                       'a11=a22=3, a_even=a_odd=4, nu11=1.5, nu22=2.5, nu_even=3.5, nu_odd=4.5, '
                       'rho=0.8. Simulation uses conditional Cholesky factorization of the corrected '
                       'covariance, without a spectral approximation. ') + common
        metadata['figure_'+str(index)] = dict(paths=paths, raw_data=str(raw_path), parameters=params,
                                              theta=angles.tolist(), seed=family_seed, p=1, u0=[1., 0.],
                                              rho_max_by_theta=rho_bounds, color_limits=[-color_limit, color_limit],
                                              lag_convention='Cov(Z1(s), Z2(t)) = C12(t-s)',
                                              validation=validation, caption=caption)
    metadata_path = outdir/'figures_1_to_3_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir/'figures_1_to_3_captions.txt').write_text('\n\n'.join(
        key.replace('_', ' ').title()+': '+item['caption'] for key, item in metadata.items()), encoding='utf-8')
    return metadata


# simulation_reporting.py
"""Publication figures and transparent Monte Carlo parameter summaries.

The simulation driver supplies a record for every attempted data set, including
failed fits.  Numerical local checks are reported separately from finite fits;
no boxplot outliers are removed and no failure is silently dropped.
"""
from pathlib import Path
import html
import json
import math
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RECOVERY_PARAMETERS = {
    'G_odd': ['a11', 'a22', 'a_odd', 'rho'],
    'G_mixed': ['a11', 'a22', 'a_even', 'a_odd', 'theta', 'rho'],
    'M_odd': ['a11', 'a22', 'a_odd', 'nu11', 'nu22', 'nu_odd', 'rho'],
    'M_mixed': ['a11', 'a22', 'a_even', 'a_odd', 'nu11', 'nu22',
                'nu_even', 'nu_odd', 'theta', 'rho'],
}
RECOVERY_FIGURES = {'G_odd': 4, 'G_mixed': 5, 'M_odd': 6, 'M_mixed': 7}
RECOVERY_NAMES = {'G_odd': 'Gaussian odd', 'G_mixed': 'Gaussian mixed',
                  'M_odd': 'Matérn odd', 'M_mixed': 'Matérn mixed'}
RECOVERY_LABELS = {
    'a11': r'$a_{11}$', 'a22': r'$a_{22}$', 'a_even': r'$a_{\rm even}$',
    'a_odd': r'$a_{\rm odd}$', 'nu11': r'$\nu_{11}$', 'nu22': r'$\nu_{22}$',
    'nu_even': r'$\nu_{\rm even}$', 'nu_odd': r'$\nu_{\rm odd}$',
    'theta': r'$\theta$ (radians)', 'rho': r'$\rho$',
    'even_amplitude': r'$\rho\cos\theta$', 'odd_amplitude': r'$\rho\sin\theta$',
}


def _recovery_validated(record):
    """Use independent local-check evidence, not an optimizer stopping flag."""
    for key in ('locally_validated', 'success'):
        if key in record:
            return bool(record[key])
    for key in ('local_diagnostics', 'diagnostics'):
        diagnostic = record.get(key) or {}
        if isinstance(diagnostic, dict) and 'passed' in diagnostic:
            return bool(diagnostic['passed'])
    return record.get('status') == 'locally_checked'


def _recovery_finite(value):
    try:
        return np.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _recovery_boundary_names(record):
    """Associate transformed-coordinate bounds with affected model parameters.

    For derived cross scales/smoothness, this means a corresponding cross gap
    constraint was active, not that the derived parameter has a fixed box bound.
    Raw bound names are retained in the complete estimate table.
    """
    mapping = {
        'log_length11': 'a11', 'log_length22': 'a22',
        'log_length_even': 'a_even', 'log_length_odd': 'a_odd',
        'gap_even': 'a_even', 'sqrt_gap_even': 'a_even',
        'log_gap_odd': 'a_odd', 'gap_odd': 'a_odd',
        'nu_gap_even': 'nu_even', 'nu_gap_odd': 'nu_odd',
        'rho_fraction': 'rho',
    }
    raw = record.get('bounds_reached') or []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, dict):
        raw = [key for key, value in raw.items() if value]
    names = set()
    for item in raw:
        if isinstance(item, dict):
            item = item.get('name', item.get('parameter', ''))
        name = re.split(r'[:= ]', str(item))[0]
        names.add(mapping.get(name, name))
    diagnostic = record.get('diagnostics') or {}
    if isinstance(diagnostic, dict):
        names.update(diagnostic.get('boundary_parameters') or [])
    return names


def _recovery_clean_json(value):
    if isinstance(value, dict):
        return {str(key): _recovery_clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_recovery_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _recovery_clean_json(value.tolist())
    if isinstance(value, np.generic):
        return _recovery_clean_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _recovery_write_json(path, value):
    Path(path).write_text(json.dumps(_recovery_clean_json(value), indent=2,
                                   ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _recovery_summarize(values, truth, boundaries, rho_values=None, true_rho=None):
    values = np.asarray(values, dtype=float)
    n = len(values)
    result = dict(n_used=n, mean=np.nan, bias=np.nan, MCSE_bias=np.nan,
                  RMSE=np.nan, median=np.nan, Q1=np.nan, Q3=np.nan, IQR=np.nan,
                  SD=np.nan, boundary_hit_fraction=np.nan,
                  rho_sign_flip_fraction=np.nan)
    if not n:
        return result
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method='linear')
    error = values-float(truth)
    sd = np.std(values, ddof=1) if n > 1 else np.nan
    result.update(mean=np.mean(values), bias=np.mean(error),
                  MCSE_bias=sd/np.sqrt(n), RMSE=np.sqrt(np.mean(error*error)),
                  median=median, Q1=q1, Q3=q3, IQR=q3-q1, SD=sd,
                  boundary_hit_fraction=np.mean(boundaries))
    if rho_values is not None and _recovery_finite(true_rho) and float(true_rho) != 0:
        rho_values = np.asarray(rho_values, dtype=float)
        rho_values = rho_values[np.isfinite(rho_values)]
        if len(rho_values):
            result['rho_sign_flip_fraction'] = np.mean(rho_values*float(true_rho) < 0)
    return result


def _recovery_latex_escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def _recovery_number(value):
    if not _recovery_finite(value):
        return '--'
    value = float(value)
    if value == 0:
        return '0'
    return '{:.4g}'.format(value)


def _recovery_latex_table(frame, columns, caption):
    heads = {'parameter': 'Parameter', 'true': 'True', 'n_attempted': 'Attempted',
             'n_finite': 'Finite', 'n_validated': 'Checked', 'n_used': 'Used',
             'mean': 'Mean', 'bias': 'Bias', 'MCSE_bias': 'MCSE(bias)',
             'RMSE': 'RMSE', 'median': 'Median', 'Q1': '$Q_1$', 'Q3': '$Q_3$',
             'IQR': 'IQR', 'SD': 'SD', 'boundary_hit_fraction': 'Bound frac.',
             'rho_sign_flip_fraction': '$\rho$ sign-flip frac.'}
    lines = [r'\begin{center}', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{longtable}{l'+'r'*(len(columns)-1)+'}',
             r'\caption{'+_recovery_latex_escape(caption)+r'}\\', r'\toprule',
             ' & '.join(heads.get(col, _recovery_latex_escape(col)) for col in columns)+r' \\',
             r'\midrule', r'\endfirsthead', r'\toprule',
             ' & '.join(heads.get(col, _recovery_latex_escape(col)) for col in columns)+r' \\',
             r'\midrule', r'\endhead']
    for row in frame.to_dict('records'):
        cells = [_recovery_latex_escape(row[col]) if col == 'parameter'
                 else _recovery_number(row[col]) for col in columns]
        lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{longtable}', r'\end{center}'])


def _recovery_render_boxplot(model, model_records, truth, path, protocol):
    parameters = RECOVERY_PARAMETERS[model]
    ncols = 2 if len(parameters) == 4 else 3
    nrows = int(np.ceil(len(parameters)/ncols))
    n_validated = sum(_recovery_validated(record) for record in model_records)
    n_complete = sum(all(_recovery_finite((record.get('params') or {}).get(name))
                         for name in parameters) for record in model_records)
    degree = protocol.get('p')
    degree_text = ', p={}'.format(degree) if degree is not None else ''
    dimensions = {(record.get('N'), record.get('R')) for record in model_records}
    design_text = ''
    if len(dimensions) == 1:
        sites, replicates = next(iter(dimensions))
        if sites is not None and replicates is not None:
            realization_text = ('1 joint field realization' if replicates == 1 else
                                '{} joint field replicates'.format(replicates))
            design_text = '{} sites; {} per dataset. '.format(sites, realization_text)
    colors = ('#236E82', '#C26E28')
    rc = {'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 12,
          'axes.labelsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
          'axes.spines.top': False, 'axes.spines.right': False}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.0*nrows+1.0),
                                 squeeze=False)
        for axis, parameter in zip(axes.flat, parameters):
            groups = [[], []]
            for record in model_records:
                value = (record.get('params') or {}).get(parameter)
                if _recovery_finite(value):
                    groups[0 if _recovery_validated(record) else 1].append(float(value))
            positions, ticklabels = [], []
            for group_index, values in enumerate(groups):
                if not values:
                    continue
                x = len(positions)+1
                positions.append(x)
                label = 'Locally checked' if group_index == 0 else 'Unvalidated'
                ticklabels.append('{}\n(n={})'.format(label, len(values)))
                axis.boxplot([values], positions=[x], widths=0.45, showfliers=True,
                             patch_artist=True,
                             boxprops=dict(facecolor=colors[group_index], alpha=0.25,
                                           edgecolor=colors[group_index]),
                             medianprops=dict(color=colors[group_index], linewidth=1.8),
                             whiskerprops=dict(color=colors[group_index]),
                             capprops=dict(color=colors[group_index]),
                             flierprops=dict(marker='o', markersize=3.2,
                                             markerfacecolor=colors[group_index],
                                             markeredgecolor=colors[group_index], alpha=0.65))
            axis.axhline(float(truth[parameter]), color='#A83232', linestyle='--',
                         linewidth=1.2, label='Generating value')
            axis.set_title(RECOVERY_LABELS[parameter])
            if positions:
                axis.set_xticks(positions)
                axis.set_xticklabels(ticklabels)
                axis.set_xlim(0.4, len(positions)+0.6)
            else:
                axis.set_xticks([])
                axis.text(0.5, 0.5, 'No finite estimates', transform=axis.transAxes,
                          ha='center', va='center')
            axis.grid(axis='y', color='#E2E2E2', linewidth=0.6)
            axis.set_axisbelow(True)
            axis.margins(y=0.10)
        for axis in list(axes.flat)[len(parameters):]:
            axis.set_visible(False)
        fig.suptitle('Figure {}. {} parameter recovery{}'.format(
            RECOVERY_FIGURES[model], RECOVERY_NAMES[model], degree_text), y=0.995, fontsize=15)
        caption = (design_text+'{} attempted; {} locally checked; {} complete finite fits; '
                   '{} without complete estimates. Dashed red: generating value. '
                   'Boxes: Q1–Q3; line: median; whiskers: 1.5 × IQR. All outliers retained.'.format(
                       len(model_records), n_validated, n_complete, len(model_records)-n_complete))
        caption = textwrap.fill(caption, width=98 if ncols == 2 else 145)
        fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9)
        footer_fraction = (caption.count('\n')+1)*0.16/fig.get_figheight()+0.035
        fig.tight_layout(rect=(0, footer_fraction, 1, 0.975))
        fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
    return dict(png=str(path.with_suffix('.png')), pdf=str(path.with_suffix('.pdf')))


def render_recovery_results(records, truths, outdir, protocol):
    """Write Figures 4–7, estimate records, and supplementary CSV/TeX/HTML.

    Parameters
    ----------
    records : list of dict
        One entry per attempted fit with ``model``, ``simulation_id``, ``seed``,
        ``params``, ``status``, and local-check evidence.  See
        ``_recovery_validated`` for supported flags.  ``bounds_reached`` contains
        raw optimizer bound names and is preserved for auditing.
    truths : dict
        Generating manuscript parameters keyed by G_odd/G_mixed/M_odd/M_mixed.
    outdir : pathlib.Path or str
        Directory for the files; existing report files are deliberately updated.
    protocol : dict
        Design and fitting settings, including replication counts and domain.

    Returns a JSON-serializable dictionary of paths and per-model counts.  The
    primary performance table conditions on passing the local numerical check;
    the all-finite table is a separate sensitivity analysis.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    models = [model for model in RECOVERY_PARAMETERS if model in truths]
    unknown = sorted(set(record.get('model') for record in records)-set(models))
    if unknown:
        raise ValueError('Recovery records have missing generating truths: '+str(unknown))
    for model in models:
        missing = [name for name in RECOVERY_PARAMETERS[model]
                   if not _recovery_finite(truths[model].get(name))]
        if missing:
            raise ValueError('Missing/nonfinite truth for {}: {}'.format(model, missing))

    primary_rows, finite_rows, coefficient_rows, estimate_rows, count_rows = [], [], [], [], []
    figures = {}
    for model in models:
        items = [record for record in records if record.get('model') == model]
        parameters, truth = RECOVERY_PARAMETERS[model], truths[model]
        n_checked = sum(_recovery_validated(record) for record in items)
        n_complete = sum(all(_recovery_finite((record.get('params') or {}).get(name))
                             for name in parameters) for record in items)
        status_counts = {}
        for record in items:
            status = str(record.get('status', 'unknown'))
            status_counts[status] = status_counts.get(status, 0)+1
            estimate = dict(model=model, simulation_id=record.get('simulation_id'),
                            seed=record.get('seed'), status=status,
                            locally_validated=_recovery_validated(record),
                            nll=record.get('nll'),
                            bounds_reached=json.dumps(_recovery_clean_json(
                                record.get('bounds_reached') or [])))
            for name in parameters:
                estimate[name] = (record.get('params') or {}).get(name)
            estimate_rows.append(estimate)
        count_rows.append(dict(model=model, n_attempted=len(items), n_complete_finite=n_complete,
                               n_locally_validated=n_checked,
                               n_without_complete_estimates=len(items)-n_complete,
                               n_unvalidated=len(items)-n_checked,
                               status_counts=json.dumps(status_counts, sort_keys=True)))

        for parameter in parameters:
            finite = [record for record in items
                      if _recovery_finite((record.get('params') or {}).get(parameter))]
            checked = [record for record in finite if _recovery_validated(record)]
            common = dict(model=model, parameter=parameter, true=float(truth[parameter]),
                          n_attempted=len(items), n_finite=len(finite), n_validated=len(checked))
            for subset, target, subset_name in ((checked, primary_rows, 'locally_validated'),
                                                (finite, finite_rows, 'all_finite')):
                values = [float(record['params'][parameter]) for record in subset]
                boundaries = [parameter in _recovery_boundary_names(record) for record in subset]
                rho_values = None
                if parameter in ('rho', 'theta'):
                    rho_values = [record['params'].get('rho', np.nan) for record in subset]
                row = dict(common, subset=subset_name)
                row.update(_recovery_summarize(values, truth[parameter], boundaries,
                                               rho_values, truth.get('rho')))
                target.append(row)
        if model.endswith('mixed'):
            for parameter, function in (('even_amplitude', np.cos), ('odd_amplitude', np.sin)):
                complete_mixing = [record for record in items
                                   if _recovery_finite((record.get('params') or {}).get('rho'))
                                   and _recovery_finite((record.get('params') or {}).get('theta'))]
                checked_mixing = [record for record in complete_mixing if _recovery_validated(record)]
                truth_coefficient = float(truth['rho']*function(truth['theta']))
                for subset, subset_name in ((checked_mixing, 'locally_validated'),
                                            (complete_mixing, 'all_finite')):
                    values = [record['params']['rho']*function(record['params']['theta'])
                              for record in subset]
                    boundaries = [bool({'rho', 'theta'} & _recovery_boundary_names(record))
                                  for record in subset]
                    row = dict(model=model, parameter=parameter, true=truth_coefficient,
                               n_attempted=len(items), n_finite=len(complete_mixing),
                               n_validated=len(checked_mixing), subset=subset_name)
                    row.update(_recovery_summarize(values, truth_coefficient, boundaries))
                    coefficient_rows.append(row)
        figures[model] = _recovery_render_boxplot(model, items, truth,
            outdir/'figure_{:02d}_{}_boxplots'.format(RECOVERY_FIGURES[model], model), protocol)

    frames = {'supplementary_parameter_summary': pd.DataFrame(primary_rows),
              'supplementary_parameter_summary_all_finite': pd.DataFrame(finite_rows),
              'supplementary_mixing_coefficients': pd.DataFrame(coefficient_rows),
              'simulation_estimates': pd.DataFrame(estimate_rows),
              'simulation_status_counts': pd.DataFrame(count_rows)}
    paths = {}
    for name, frame in frames.items():
        path = outdir/(name+'.csv')
        frame.to_csv(path, index=False, float_format='%.12g')
        paths[name] = str(path)
    _recovery_write_json(outdir/'simulation_records.json', records)
    _recovery_write_json(outdir/'simulation_protocol.json', protocol)

    notes = [
        'Each record corresponds to an attempted simulated data set. Primary summaries use only '
        'estimates passing independent local numerical checks; these checks do not establish a global optimum. '
        'Counts and the separate all-finite sensitivity tables expose exclusions. Conditional summaries can '
        'be affected by selective fitting failures.',
        'Bias is mean(estimate minus generating value); RMSE is the square root of the mean squared error. '
        'SD is the sample standard deviation (denominator n minus 1), and MCSE(bias) is SD/sqrt(n), '
        'assuming independent simulation data sets. Q1, median and Q3 use linearly interpolated sample '
        'quantiles. IQR is Q3 minus Q1. Undefined quantities are shown as dashes.',
        'All finite estimates, including boxplot outliers and unvalidated fits, are retained in '
        'simulation_estimates.csv and simulation_records.json. Boxplots separate validated and unvalidated '
        'fits; boxes span Q1 to Q3, whiskers extend to the most extreme values within 1.5 IQR, and '
        'points beyond the whiskers remain visible. No axis limits truncate the estimates.',
        'The boundary fraction measures activation of the corresponding optimizer constraint. '
        'For a derived cross scale or smoothness it refers to its cross-gap constraint; for rho it refers '
        'to the spectral-validity fraction constraint. Raw bound names are preserved for inspection. '
        'It is not a test for singularity of the covariance matrix.',
        'Mixed models use signed rho and canonical theta in [0, pi]. Theta summaries use ordinary '
        'differences in radians, not circular error. Near a representation boundary or rho=0, raw angle '
        'and sign summaries can be misleading; the additional rho*cos(theta) and rho*sin(theta) tables '
        'describe the identifiable even/odd mixing coefficients. Sign-flip fractions compare the fitted '
        'rho sign with its nonzero generating value and are only shown for rho and theta.',
        'The fitted parameter space uses the sufficient spectral-validity region specified in the '
        'protocol. The mixed validity bound may be conservative. All model parameters and reported '
        'scale units follow the simulation domain; rho is the spectral amplitude and is not generally '
        'a collocated correlation.',
    ]
    summary_columns = ['parameter', 'true', 'n_attempted', 'n_finite', 'n_validated',
                       'n_used', 'mean', 'bias', 'MCSE_bias', 'RMSE', 'SD']
    distribution_columns = ['parameter', 'true', 'n_used', 'Q1', 'median', 'Q3', 'IQR',
                            'boundary_hit_fraction', 'rho_sign_flip_fraction']
    tex = [r'\documentclass[10pt]{article}', r'\usepackage[T1]{fontenc}',
           r'\usepackage[utf8]{inputenc}', r'\usepackage[margin=16mm,a4paper,landscape]{geometry}',
           r'\usepackage{booktabs,longtable,amsmath}',
           r'\title{Supplementary Monte Carlo parameter-recovery results}',
           r'\author{}', r'\date{}', r'\begin{document}', r'\maketitle',
           '\n\n'.join(_recovery_latex_escape(note) for note in notes),
           r'\section*{Simulation and fitting protocol}', r'\begin{itemize}']
    for key, value in protocol.items():
        value = json.dumps(_recovery_clean_json(value), ensure_ascii=True) if isinstance(value, (dict, list)) else value
        tex.append(r'\item \textbf{'+_recovery_latex_escape(key)+r':} '+_recovery_latex_escape(value))
    tex.extend([r'\end{itemize}', r'\clearpage'])
    html_parts = ['<!doctype html><html><head><meta charset="utf-8">',
                  '<title>Supplementary simulation results</title><style>',
                  'body{max-width:1500px;margin:35px auto;padding:0 24px;font:15px/1.5 system-ui,sans-serif;color:#172027}',
                  'h1,h2,h3{line-height:1.2}table{border-collapse:collapse;font-size:12px;width:100%}',
                  'th,td{border-bottom:1px solid #ddd;padding:6px 8px;text-align:right;white-space:nowrap}',
                  'th{background:#edf3f5}td:first-child,th:first-child{text-align:left}',
                  '.table{overflow-x:auto}img{width:100%;max-width:1200px}pre{white-space:pre-wrap;background:#f3f5f6;padding:15px}',
                  '</style></head><body><h1>Supplementary Monte Carlo parameter-recovery results</h1>',
                  ''.join('<p>'+html.escape(note)+'</p>' for note in notes),
                  '<h2>Protocol</h2><pre>'+html.escape(json.dumps(_recovery_clean_json(protocol), indent=2))+'</pre>',
                  '<h2>Attempt and fitting counts</h2><div class="table">',
                  frames['simulation_status_counts'].to_html(index=False, escape=True), '</div>']
    for model in models:
        tex.append(r'\section*{'+_recovery_latex_escape(RECOVERY_NAMES[model])+r'}')
        counts = next(row for row in count_rows if row['model'] == model)
        tex.append(_recovery_latex_escape(
            '{} attempted datasets; {} complete finite fits; {} passed the local numerical check; '
            '{} lacked complete estimates; {} were unvalidated. Status counts: {}.'.format(
                counts['n_attempted'], counts['n_complete_finite'], counts['n_locally_validated'],
                counts['n_without_complete_estimates'], counts['n_unvalidated'], counts['status_counts'])))
        html_parts.append('<h2>Figure {}: {}</h2><img src="{}" alt="{} parameter recovery">'.format(
            RECOVERY_FIGURES[model], html.escape(RECOVERY_NAMES[model]),
            html.escape(Path(figures[model]['png']).name), html.escape(RECOVERY_NAMES[model])))
        for table_name, title in (('supplementary_parameter_summary', 'Primary: locally checked fits'),
                                  ('supplementary_parameter_summary_all_finite', 'Sensitivity: all finite estimates')):
            frame = frames[table_name]
            frame = frame[frame['model'] == model]
            tex.append(_recovery_latex_table(frame, summary_columns,
                       RECOVERY_NAMES[model]+'. '+title+'. Accuracy and usable sample sizes.'))
            tex.append(_recovery_latex_table(frame, distribution_columns,
                       RECOVERY_NAMES[model]+'. '+title+'. Distribution and boundary diagnostics.'))
            html_parts.extend(['<h3>'+html.escape(title)+'</h3><div class="table">',
                frame.drop(columns=['model', 'subset']).to_html(index=False, escape=True,
                                                               float_format=_recovery_number, na_rep='--'),
                '</div>'])
        coefficients = frames['supplementary_mixing_coefficients']
        if not coefficients.empty:
            for subset in ('locally_validated', 'all_finite'):
                frame = coefficients[(coefficients['model'] == model) & (coefficients['subset'] == subset)]
                if frame.empty:
                    continue
                tex.append(_recovery_latex_table(frame, summary_columns,
                           RECOVERY_NAMES[model]+'. Mixing coefficients; '+subset+'.'))
                html_parts.extend(['<h3>Mixing coefficients: '+html.escape(subset)+'</h3><div class="table">',
                    frame.drop(columns=['model', 'subset']).to_html(index=False, escape=True,
                                                                   float_format=_recovery_number, na_rep='--'),
                    '</div>'])
        tex.append(r'\clearpage')
    tex.append(r'\end{document}')
    tex_path = outdir/'supplementary_tables.tex'
    tex_path.write_text('\n'.join(tex)+'\n', encoding='utf-8')
    html_path = outdir/'supplementary_tables.html'
    html_path.write_text('\n'.join(html_parts+['</body></html>']), encoding='utf-8')
    paths.update(supplementary_tex=str(tex_path), supplementary_html=str(html_path),
                 simulation_records=str(outdir/'simulation_records.json'),
                 simulation_protocol=str(outdir/'simulation_protocol.json'))
    manifest = dict(figures=figures, tables=paths, counts=count_rows)
    _recovery_write_json(outdir/'recovery_report_manifest.json', manifest)
    return manifest


# experiment_driver.py
"""CLI and reproducible protocols for the corrected paper experiments."""
import argparse
import hashlib
import json
import os
import platform
import html
import shutil
import zipfile
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import numpy as np
import scipy
from scipy.linalg import cholesky, solve_triangular

RECOVERY_MODELS = ('G_odd','G_mixed','M_odd','M_mixed')


def recovery_truths():
    return {
        'G_odd': dict(a11=3.,a22=3.,a_odd=4.,rho=.6),
        'G_mixed': dict(a11=3.,a22=3.,a_even=4.,a_odd=4.,theta=np.pi/4,rho=.6),
        'M_odd': dict(a11=3.,a22=3.,a_odd=4.,nu11=1.5,nu22=2.5,nu_odd=3.5,rho=.6),
        'M_mixed': dict(a11=3.,a22=3.,a_even=4.,a_odd=4.,nu11=1.5,nu22=2.5,
                        nu_even=3.5,nu_odd=4.5,theta=np.pi/4,rho=.6),
    }


def certify_truth(model, values, p=1):
    family,kind=model.split('_')
    params=values.copy()
    theta=params.get('theta', np.pi/2 if kind=='odd' else 0.)
    we=0. if kind=='odd' else float(np.cos(theta))
    wo=0. if kind=='even' else float(np.sin(theta))
    terms=[]
    for part,weight in [('even',we),('odd',wo)]:
        if weight==0:
            continue
        log_m=log_component_sup('gaussian' if family=='G' else 'matern',
             params['a11'],params['a22'],params['a_'+part],
             nu11=params.get('nu11'),nu22=params.get('nu22'),nuc=params.get('nu_'+part),
             p=p,odd=(part=='odd'))
        terms.append(2*np.log(abs(weight))+log_m)
    log_bound=float(np.logaddexp.reduce(terms))
    rho_max=float(np.exp(-.5*log_bound))
    ratio=float((params['rho']/rho_max)**2)
    if not np.isfinite(ratio) or ratio>=1:
        raise ValueError('Invalid generating parameters for '+model)
    params.update(theta=float(theta),weight_even=we,weight_odd=wo,rho_max=rho_max,
                  rho_fraction=params['rho']/rho_max,spectral_ratio_bound=ratio)
    return params


def recovery_config(model, settings, seed):
    return FitConfig(seed=int(seed),n_candidates=settings['candidates'],n_starts=settings['starts'],
        maxiter=settings['maxiter'],ftol=1e-10,
        length_bounds=(.5,10.) if model.startswith('G') else (1/15.,2.),
        marginal_nu_bounds=(.4,30.),cross_nu_gap_max=30.,
        gaussian_gap_max=20.,gaussian_odd_gap_min=1e-3,fraction_limit=.995,
        mean_mode='zero',marginal_sds=(1.,1.),nugget_variance=0.,blas_threads=1)


def prepare_recovery(settings):
    grid=np.linspace(0,10,8)
    xx,yy=np.meshgrid(grid,grid)
    locations=np.column_stack([xx.ravel(),yy.ravel()])
    geometry=Geometry(locations,0.)
    prepared={}
    for name,values in recovery_truths().items():
        family,kind=name.split('_')
        spec=ModelSpec(family,kind,1)
        params=certify_truth(name,values)
        covariance=covariance_matrix(spec,params,geometry)
        factor=cholesky(covariance,lower=True,check_finite=True)
        eig=np.linalg.eigvalsh(covariance)
        prepared[name]=dict(spec=spec,params=params,factor=factor,
            minimum_eigenvalue=float(eig[0]),condition_number=float(eig[-1]/eig[0]))
    return geometry,prepared


_EXPERIMENT_WORKER={}
def initialize_experiment_worker(settings):
    geometry,prepared=prepare_recovery(settings)
    _EXPERIMENT_WORKER.update(settings=settings,geometry=geometry,prepared=prepared)


def run_recovery_task(task):
    model,simulation_id=task
    settings=_EXPERIMENT_WORKER['settings']
    geo=_EXPERIMENT_WORKER['geometry']
    prepared=_EXPERIMENT_WORKER['prepared'][model]
    index=RECOVERY_MODELS.index(model)
    seed_sequence=np.random.SeedSequence([settings['seed'],index,simulation_id])
    data_seed,fit_seed=seed_sequence.spawn(2)
    rng=np.random.default_rng(data_seed)
    observations=prepared['factor']@rng.standard_normal((2*geo.n,settings['replicates']))
    seed=int(fit_seed.generate_state(1)[0])
    config=recovery_config(model,settings,seed)
    started=time.perf_counter()
    try:
        result=robust_fit_model(prepared['spec'],geo,observations,config=config)
    except Exception as exc:
        result=dict(model=model,p=1,status='failed',converged=False,nll=None,params={},
                    error=type(exc).__name__+': '+str(exc),seconds=time.perf_counter()-started)
    white=solve_triangular(prepared['factor'],observations,lower=True,check_finite=False)
    true_nll=.5*(np.sum(white*white)+settings['replicates']*(
        2*np.log(np.diag(prepared['factor'])).sum()+2*geo.n*np.log(2*np.pi)))
    finite=result.get('nll') is not None and np.isfinite(result['nll'])
    result.update(simulation_id=simulation_id,seed=seed,
        data_seed_state=data_seed.generate_state(4).tolist(),
        locally_validated=bool(result.get('converged',False)),
        true_nll=float(true_nll),nll_minus_true=float(result['nll']-true_nll) if finite else None,
        observations_sha256=hashlib.sha256(observations.tobytes()).hexdigest(),
        N=geo.n,R=settings['replicates'])
    return result


def run_refinement_task(baseline):
    model=baseline['model']
    settings=_EXPERIMENT_WORKER['settings']
    geo=_EXPERIMENT_WORKER['geometry']
    prepared=_EXPERIMENT_WORKER['prepared'][model]
    sequence=np.random.SeedSequence([settings['seed'],RECOVERY_MODELS.index(model),baseline['simulation_id']])
    data_seed,_=sequence.spawn(2)
    rng=np.random.default_rng(data_seed)
    observations=prepared['factor']@rng.standard_normal((2*geo.n,settings['replicates']))
    assert hashlib.sha256(observations.tobytes()).hexdigest()==baseline['observations_sha256']
    config=recovery_config(model,settings,baseline['seed'])
    result=refine_gaussian_fit(prepared['spec'],geo,observations,baseline,config)
    result['locally_validated']=bool(result.get('converged',False))
    result['nll_minus_true']=result['nll']-baseline['true_nll'] if result.get('nll') is not None else None
    return result


def experiment_protocol(settings,prepared):
    return dict(title='Corrected covariance paper experiments',
        settings=settings,grid='8 x 8 equally spaced on [0,10]^2',p=1,
        direction_degrees=0,means=[0.,0.],marginal_sills=[1.,1.],nugget=0.,
        repeated_sampling='Each dataset contains one joint bivariate field realization by default, as requested.',
        initialization='Data-independent reproducible multistarts; no starts centred on generating parameters.',
        refinement_policy='Every Gaussian trial receives the same additional fitted-marginal, moderate-cross-gap and both-amplitude-sign searches; independent of the likelihood at the truth.',
        covariance_validity='Continuous-frequency component bounds; sufficient mixed bound; no eigenvalue clipping.',
        simulation_method='Exact finite-site joint Gaussian Cholesky draws for Figures 4-7.',
        original_design=dict(datasets_per_model=50,replicates_per_dataset=20,sites=64),
        truths=recovery_truths(),
        generating_validity={name:dict({k:v for k,v in item.items() if k in ['minimum_eigenvalue','condition_number']},
             rho_max=item['params']['rho_max'],spectral_ratio_bound=item['params']['spectral_ratio_bound'])
             for name,item in prepared.items()},
        fit_configurations={name:asdict(recovery_config(name,settings,settings['seed'])) for name in RECOVERY_MODELS},
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__)


def write_experiment_readme(out,records,protocol):
    settings=protocol['settings']
    text=['# Corrected covariance paper experiments',
          '## Repeated-simulation design',
          '{} independent datasets per model, each containing **{} bivariate field realization(s)** '
          'on 64 sites. Four models give {} attempted fits. The degree is p=1; direction is (1,0); '
          'means are zero, sills are one and the nugget is zero.'.format(
              settings['simulations'],settings['replicates'],len(records)),
          'Generating parameters for Figures 4-7 retain the manuscript values and satisfy the '
          'corrected spectral bounds. Estimation uses the documented bounded, sufficient validity region. '
          'All fits use reproducible starts independent of the true parameters. Every Gaussian trial '
          'also receives the same second-stage sign/quadrant searches. Original fits remain in initial_fit.',
          '| Model | Attempted | Locally checked | Finite but not checked | At least one search bound |',
          '|---|---:|---:|---:|---:|']
    for name in RECOVERY_MODELS:
        group=[r for r in records if r['model']==name]
        passed=sum(bool(r.get('locally_validated')) for r in group)
        finite=sum(r.get('nll') is not None for r in group)
        text.append('| {} | {} | {} | {} | {} |'.format(
            name,len(group),passed,finite-passed,sum(bool(r.get('bounds_reached')) for r in group)))
    text+=['Local numerical checks are not global-optimum certificates. Boxplots retain all finite '
           'estimates and all outliers, distinguishing unvalidated fits. Primary tables use locally '
           'checked fits; all-finite tables disclose the sensitivity to exclusions. Boundary rates '
           'and weakly identified mixing parameters must be considered when interpreting recovery.',
           '## Files',
           '- Figures 1-3: `figures/`, with PNG and PDF versions; actual angular values and fields in `raw_data/`.',
           '- Figures 4-7: `figure_04_G_odd_boxplots`, `figure_05_G_mixed_boxplots`, '
           '`figure_06_M_odd_boxplots`, `figure_07_M_mixed_boxplots`, each PNG and PDF.',
           '- `supplementary_tables.html`: readable tables and boxplots.',
           '- `supplementary_tables.tex`: standalone LaTeX tables.',
           '- `supplementary_parameter_summary.csv`: bias, RMSE, MCSE of bias, median, Q1, Q3, IQR, SD and counts.',
           '- `supplementary_parameter_summary_all_finite.csv`: corresponding all-finite summaries.',
           '- `supplementary_mixing_coefficients.csv`: even/odd coefficient summaries.',
           '- `simulation_estimates.csv`, `simulation_records.json`, `raw_fits/`: every trial and its diagnostics.',
           '- `experiment_protocol.json`: true parameters, bounds, seeds, software and validity checks.',
           '## Figure corrections and captions',
           'Figure 1 specifies compatible marginals and holds a valid amplitude fixed across p=1,3,5: '
           'Gaussian a11=a22=0.5, a_odd=1, rho=0.004; Matérn a11=a22=a_odd=1, '
           'nu11=nu22=1, nu_odd=4, rho=0.7. Lag length is one. Panel vertical scales vary.',
           'Figure 2 uses rho=0.5 (the previous 0.85 violates validity at theta=0). '
           'Figure 3 retains rho=0.8. All other field parameters match the manuscript. '
           'The first field and latent innovations are shared across theta. Bicubic interpolation '
           'only affects display. Figure 2 uses validated spectral quadrature; Figure 3 uses exact '
           'conditional Cholesky. Detailed captions and numerical errors are in figures_1_to_3_captions.txt '
           'and figures_1_to_3_metadata.json.',
           'Figure 4: Gaussian odd recovery, (a11,a22,a_odd,rho)=(3,3,4,0.6).',
           'Figure 5: Gaussian mixed recovery, (a11,a22,a_even,a_odd,theta,rho)=(3,3,4,4,pi/4,0.6).',
           'Figure 6: Matérn odd recovery, (a11,a22,a_odd,nu11,nu22,nu_odd,rho)=(3,3,4,1.5,2.5,3.5,0.6).',
           'Figure 7: Matérn mixed recovery, scales=(3,3,4,4), smoothness=(1.5,2.5,3.5,4.5), theta=pi/4, rho=0.6.',
           'The manuscript inference claims require reassessment using these new distributions. '
           'A single realization on the original grid provides limited information about short-range '
           'Matérn parameters; wide intervals and boundary estimates must not be removed.',
           '## Run again',
           '```bash\npython corrected_paper_experiments.py --output new_results --simulations {} '
           '--replicates {} --workers {}\n```'.format(settings['simulations'],settings['replicates'],settings['workers']),
           'Add `--resume` to continue an interrupted run using identical settings. '
           'Use `--stage summaries` to rebuild only the recovery plots and tables. '
           'No notebook or auxiliary Python module is required.',
           'Dependencies: Python >=3.8, NumPy, SciPy >=1.10, pandas >=2.0, Matplotlib, threadpoolctl.']
    paragraphs=[]
    for entry in text:
        if entry.startswith('|') and paragraphs and paragraphs[-1].startswith('|'):
            paragraphs[-1]+='\n'+entry
        else:
            paragraphs.append(entry)
    (out/'README.md').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
    items=[('Figure 1: effect of p','figures/figure_1_effect_of_p.png'),
           ('Figure 2: Gaussian fields','figures/figure_2_gaussian_fields.png'),
           ('Figure 3: Matérn fields','figures/figure_3_matern_fields.png')]
    items += [('Figure {}: {}'.format(4+i,name),
               'figure_{:02d}_{}_boxplots.png'.format(4+i,name)) for i,name in enumerate(RECOVERY_MODELS)]
    page=['<!doctype html><html><meta charset="utf-8"><title>Corrected paper experiments</title>',
          '<style>body{max-width:1150px;margin:30px auto;font:16px/1.5 system-ui;padding:0 20px}img{width:100%}a{color:#23648a}</style>',
          '<h1>Corrected paper experiments</h1>',
          '<p>{} independent trials per model; {} joint realization per trial; 64 sites.</p>'.format(settings['simulations'],settings['replicates']),
          '<p><a href="supplementary_tables.html">Supplementary tables and fitting counts</a> | '
          '<a href="README.md">Protocol and interpretation</a> | '
          '<a href="corrected_paper_experiments.py">Standalone Python script</a></p>']
    page += ['<h2>{}</h2><a href="{}"><img src="{}" alt="{}"></a>'.format(
        html.escape(title),path,path,html.escape(title)) for title,path in items]
    (out/'index.html').write_text('\n'.join(page)+'</html>',encoding='utf-8')
    source=Path(__file__).resolve()
    if source != out/'corrected_paper_experiments.py':
        shutil.copy2(source,out/'corrected_paper_experiments.py')
    (out/'requirements.txt').write_text('numpy\nscipy>=1.10\npandas>=2.0\nmatplotlib\nthreadpoolctl\n',encoding='utf-8')
    archive=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zipped:
        for file in sorted(out.rglob('*')):
            if file.is_file():
                zipped.write(file,str(Path(out.name)/file.relative_to(out)))
    print('Packaged:',archive,flush=True)


def main_experiments():
    parser=argparse.ArgumentParser(description='Recreate corrected paper Figures 1-7 and Monte Carlo tables.')
    parser.add_argument('--output',type=Path,default=Path('paper_experiment_results'))
    parser.add_argument('--simulations',type=int,default=200)
    parser.add_argument('--replicates',type=int,default=1)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20260912)
    parser.add_argument('--candidates',type=int,default=32)
    parser.add_argument('--starts',type=int,default=3)
    parser.add_argument('--maxiter',type=int,default=300)
    parser.add_argument('--stage',choices=['all','recovery','figures','refinement','summaries'],default='all')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.simulations,args.replicates,args.workers,args.candidates,args.starts,args.maxiter)<1:
        parser.error('Counts must be positive.')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    settings={key:getattr(args,key) for key in ['simulations','replicates','workers','seed','candidates','starts','maxiter']}
    geometry,prepared=prepare_recovery(settings)
    protocol=experiment_protocol(settings,prepared)
    protocol['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol_file=out/'experiment_protocol.json'
    if protocol_file.exists():
        previous=json.loads(protocol_file.read_text())
        if previous.get('settings')!=settings:
            raise ValueError('Existing output uses a different protocol; choose a new output directory.')
        if args.stage in ('all','recovery') and not args.resume:
            raise FileExistsError('Use --resume or a new output directory.')
        if previous.get('script_sha256')!=protocol['script_sha256']:
            protocol['previous_script_sha256']=previous.get('previous_script_sha256',previous.get('script_sha256'))
    protocol_file.write_text(json.dumps(_json_safe(protocol),indent=2),encoding='utf-8')
    print('Output:',out,flush=True)
    if args.stage in ('all','figures'):
        metadata=generate_figures_1_to_3(out,seed=args.seed)
        print('Figures 1-3 generated.',flush=True)
    records=[]
    raw=out/'raw_fits'; raw.mkdir(exist_ok=True)
    for file in sorted(raw.glob('*.json')):
        records.append(json.loads(file.read_text()))
    if args.stage in ('all','recovery'):
        completed={(r['model'],r['simulation_id']) for r in records}
        tasks=[(model,i) for i in range(args.simulations) for model in RECOVERY_MODELS
               if (model,i) not in completed]
        started=time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_experiment_worker,
                                 initargs=(settings,)) as executor:
            futures={executor.submit(run_recovery_task,task):task for task in tasks}
            for future in as_completed(futures):
                model,simulation_id=futures[future]
                result=future.result()
                file=raw/('{}_{:04d}.json'.format(model,simulation_id))
                file.write_text(json.dumps(_json_safe(result),indent=2),encoding='utf-8')
                records.append(result)
                done=len(records)
                if done%8==0 or done==4*args.simulations:
                    passed=sum(r.get('locally_validated',False) for r in records)
                    print('{}/{} fits; {} locally checked; {:.1f} minutes elapsed'.format(
                        done,4*args.simulations,passed,(time.perf_counter()-started)/60),flush=True)
    if args.stage in ('all','refinement'):
        pending=[r for r in records if r['model'].startswith('G') and not r.get('refinement')]
        if pending:
            print('Uniform Gaussian starting-point refinement: {} trials.'.format(len(pending)),flush=True)
            with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_experiment_worker,
                                     initargs=(settings,)) as executor:
                futures=[executor.submit(run_refinement_task,r) for r in pending]
                for index,future in enumerate(as_completed(futures),1):
                    result=future.result()
                    (raw/('{}_{:04d}.json'.format(result['model'],result['simulation_id']))).write_text(
                        json.dumps(_json_safe(result),indent=2),encoding='utf-8')
                    if index%40==0 or index==len(pending):
                        print('Refined {}/{} Gaussian trials.'.format(index,len(pending)),flush=True)
            records=[json.loads(file.read_text()) for file in sorted(raw.glob('*.json'))]
    if args.stage in ('all','refinement','summaries'):
        if len(records)!=4*args.simulations:
            raise ValueError('Requested repetitions are incomplete; resume recovery first.')
        artifacts=render_recovery_results(records,recovery_truths(),out,protocol)
        write_experiment_readme(out,records,protocol)
        print('Figures 4-7 and supplementary tables generated.',flush=True)
    print('Completed stage:',args.stage,flush=True)


if __name__=='__main__':
    main_experiments()
