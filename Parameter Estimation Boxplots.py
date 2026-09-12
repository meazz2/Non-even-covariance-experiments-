from __future__ import annotations
import os
import sys
from pathlib import Path
for _thread_variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_thread_variable, '1')
_native_dll_handles = []
if sys.platform == 'win32':
    _conda_native = Path(sys.prefix)/'Library/bin'
    if _conda_native.is_dir():
        os.environ['PATH'] = str(_conda_native)+os.pathsep+os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            _native_dll_handles.append(os.add_dll_directory(str(_conda_native)))
import matplotlib
matplotlib.use('Agg')


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


























def truth_started_fit(spec, geometry, observations, config=None, provided_starts=None,
                     optimizer_options=None):
    """Optimize every supplied near-truth start; then independently check the selected fit."""
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
        candidates = [np.asarray(x, dtype=float) for x in provided_starts]
        if len(candidates) != config.n_starts:
            raise ValueError('Exactly n_starts supplied starts are required.')
        for x in candidates:
            space.decode(x)
            if not np.isfinite(problem.evaluate(x)[0]):
                raise ValueError('A supplied start has a nonfinite likelihood.')

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

        for idx, candidate in enumerate(candidates):
            run_local(objective.to_unit(candidate), 'truth_start_'+str(idx))
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
        optimizer='three supplied truth-informed starts; adaptive derivatives and local probes')
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








from pathlib import Path
import html
import json
import math
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib

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
    if not np.isfinite(rho_max) or rho_max <= 0:
        raise ValueError('No finite positive spectral amplitude bound.')
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








"""Near-truth initialization, checkpointed fits, and successful-only reporting."""
import traceback


def physical_to_native(space, physical):
    """Invert the admissible parameterization; reject instead of projecting."""
    p = certify_truth(space.spec.name, physical, space.spec.p)
    names = {}
    if space.spec.family == 'G':
        names.update(log_length11=np.log(p['a11']), log_length22=np.log(p['a22']))
        base = (p['a11']**2+p['a22']**2)/2
        if space.spec.has_even:
            names['gap_even'] = p['a_even']**2/base-1
        if space.spec.has_odd:
            gap = p['a_odd']**2/base-1
            if gap <= 0:
                raise ValueError('Gaussian odd range has no positive spectral gap.')
            names['log_gap_odd'] = np.log(gap)
    else:
        for part in ('11', '22', 'even', 'odd'):
            key = 'a'+part if part in ('11', '22') else 'a_'+part
            if key in p:
                names['log_length'+part if part in ('11', '22') else 'log_length_'+part] = -np.log(p[key])
        names.update(nu11=p['nu11'], nu22=p['nu22'])
        average = (p['nu11']+p['nu22'])/2
        if space.spec.has_even:
            names['nu_gap_even'] = p['nu_even']-average
        if space.spec.has_odd:
            names['nu_gap_odd'] = p['nu_odd']-average-space.spec.p/2
    names.update(theta=p['theta'], rho_fraction=p['rho_fraction'])
    x = np.array([names[name] for name in space.names])
    decoded = space.decode(x)  # checks every configured bound and the spectral restrictions
    for key, value in physical.items():
        if not np.isclose(decoded[key], value, rtol=2e-12, atol=1e-13):
            raise ValueError('Physical-start round trip changed '+key)
    return x


def sample_truth_starts(spec, geometry, config, true_values):
    """Independent signed uniform 10-20% changes, conditional on joint validity."""
    space = ParameterSpace(spec, geometry, config)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 102020]))
    keys = RECOVERY_PARAMETERS[spec.name]
    vectors, audit = [], []
    rejected = {}
    draws = 0
    while len(vectors) < config.n_starts and draws < 10000:
        draws += 1
        errors = rng.choice([-1., 1.], len(keys))*rng.uniform(.10, .20, len(keys))
        physical = {key: float(true_values[key]*(1+delta)) for key, delta in zip(keys, errors)}
        try:
            x = physical_to_native(space, physical)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            reason = type(exc).__name__+': '+str(exc)
            rejected[reason] = rejected.get(reason, 0)+1
            continue
        vectors.append(x)
        audit.append(dict(start_index=len(vectors)-1, draw_index=draws,
                          physical_parameters=physical,
                          relative_errors=dict(zip(keys, errors.tolist())),
                          native_parameters=dict(zip(space.names, x.tolist()))))
    if len(vectors) != config.n_starts:
        raise RuntimeError('Could not sample enough admissible 10-20% starts.')
    return vectors, dict(starts=audit, total_draws=draws, rejected_draws=sum(rejected.values()),
                         rejection_reasons=rejected,
                         rule='true_j*(1+sign_j*Uniform(0.10,0.20)); whole-vector validity rejection')


def run_truth_task(task):
    model, simulation_id = task
    settings = _EXPERIMENT_WORKER['settings']
    geo = _EXPERIMENT_WORKER['geometry']
    prepared = _EXPERIMENT_WORKER['prepared'][model]
    sequence = np.random.SeedSequence([settings['seed'], RECOVERY_MODELS.index(model), simulation_id])
    data_seed, fit_seed = sequence.spawn(2)
    observations = prepared['factor'] @ np.random.default_rng(data_seed).standard_normal((2*geo.n, 1))
    seed = int(fit_seed.generate_state(1)[0])
    config = recovery_config(model, settings, seed)
    started = time.perf_counter()
    start_audit = None
    try:
        starts, start_audit = sample_truth_starts(prepared['spec'], geo, config, recovery_truths()[model])
        result = truth_started_fit(prepared['spec'], geo, observations, config=config, provided_starts=starts)
    except Exception as exc:
        result = dict(model=model, status='failed', converged=False, nll=None, params={},
                      error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc(),
                      seconds=time.perf_counter()-started)
    white = solve_triangular(prepared['factor'], observations, lower=True, check_finite=False)
    true_nll = .5*(np.sum(white*white)+2*np.log(np.diag(prepared['factor'])).sum()+2*geo.n*np.log(2*np.pi))
    finite = result.get('nll') is not None and np.isfinite(result['nll'])
    result.update(simulation_id=simulation_id, seed=seed, start_audit=start_audit,
                  data_seed_state=data_seed.generate_state(4).tolist(),
                  locally_validated=bool(result.get('converged', False)),
                  true_nll=float(true_nll), nll_minus_true=float(result['nll']-true_nll) if finite else None,
                  observations_sha256=hashlib.sha256(observations.tobytes()).hexdigest(), N=geo.n, R=1)
    return result


def retained_fit(record):
    params = record.get('params') or {}
    return bool(record.get('converged') and (record.get('local_diagnostics') or {}).get('passed')
                and record.get('nll') is not None and np.isfinite(record['nll'])
                and all(key in params and np.isfinite(params[key]) for key in RECOVERY_PARAMETERS[record['model']])
                and np.isfinite(params.get('spectral_ratio_bound', np.nan))
                and params['spectral_ratio_bound'] < 1)


def describe_estimates(values, truth):
    values = np.asarray(values, dtype=float)
    n = len(values)
    result = dict(n_retained=n, mean=np.nan, bias=np.nan, RMSE=np.nan, median=np.nan,
                  Q1=np.nan, Q3=np.nan, IQR=np.nan, SD=np.nan, MCSE_bias=np.nan)
    if n:
        q1, median, q3 = np.quantile(values, [.25, .5, .75], method='linear')
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        result.update(mean=float(values.mean()), bias=float(values.mean()-truth),
                      RMSE=float(np.sqrt(np.mean((values-truth)**2))), median=median,
                      Q1=q1, Q3=q3, IQR=q3-q1, SD=sd, MCSE_bias=sd/np.sqrt(n))
    return result


def plot_successful_model(out, model, group, attempted):
    keys = RECOVERY_PARAMETERS[model]
    ncols = 2 if len(keys) == 4 else 3
    nrows = math.ceil(len(keys)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.65*ncols, 2.9*nrows+.65), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        values = [r['params'][key] for r in group]
        if values:
            ax.boxplot([values], positions=[1], widths=.42, patch_artist=True, whis=1.5,
                       boxprops=dict(facecolor='#b7d8ec', edgecolor='#28658c', linewidth=1.3),
                       medianprops=dict(color='#123d59', linewidth=1.8),
                       whiskerprops=dict(color='#28658c'), capprops=dict(color='#28658c'),
                       flierprops=dict(marker='o', markersize=3, markerfacecolor='#28658c',
                                       markeredgecolor='none', alpha=.5))
        ax.axhline(recovery_truths()[model][key], color='#b74435', linestyle='--', linewidth=1.35,
                   label='True value')
        ax.set_title(RECOVERY_LABELS[key], fontsize=12)
        ax.set_xticks([1]); ax.set_xticklabels(['Successful fits'])
        ax.set_xlim(.45, 1.55)
        ax.grid(axis='y', color='#e3e8eb', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    number = RECOVERY_FIGURES[model]
    fig.suptitle('Figure {}. {} parameter recovery'.format(number, RECOVERY_NAMES[model]),
                 fontsize=15, y=.993)
    fig.text(.5, .025, '{} retained / {} attempted; {} excluded. One realization per dataset; 64 sites.\n'
             'Three valid starts, each parameter 10–20% from truth. Dashed red: truth. All successful-fit outliers shown.'
             .format(len(group), attempted, attempted-len(group)), ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=[0, .105 if nrows <= 2 else .08, 1, .965])
    base = 'figure_{:02d}_{}_boxplots'.format(number, model)
    fig.savefig(out/(base+'.png'), dpi=220, bbox_inches='tight')
    fig.savefig(out/(base+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def write_success_reports(out, records, protocol, baseline=None):
    records = sorted(records, key=lambda r: (RECOVERY_MODELS.index(r['model']), r['simulation_id']))
    expected = protocol['settings']['simulations']*4
    if len(records) != expected or len({(r['model'], r['simulation_id']) for r in records}) != expected:
        raise ValueError('Reports require one checkpoint for every attempted dataset.')
    summary_rows, count_rows, estimates, excluded, start_rows, paired = [], [], [], [], [], []
    for model in RECOVERY_MODELS:
        group = [r for r in records if r['model'] == model]
        good = [r for r in group if retained_fit(r)]
        count_rows.append(dict(model=model, attempted=len(group), retained=len(good), excluded=len(group)-len(good),
                               retained_with_any_active_bound=sum(bool(r.get('bounds_reached')) for r in good)))
        for key in RECOVERY_PARAMETERS[model]:
            row = dict(model=model, parameter=key, true_value=recovery_truths()[model][key],
                       n_attempted=len(group), n_excluded=len(group)-len(good))
            row.update(describe_estimates([r['params'][key] for r in good], row['true_value']))
            row['boundary_hit_fraction'] = (np.mean([key in _recovery_boundary_names(r) for r in good]) if good else np.nan)
            summary_rows.append(row)
        plot_successful_model(out, model, good, len(group))
        for r in group:
            keep = retained_fit(r)
            diagnostic = r.get('local_diagnostics') or {}
            if keep:
                estimates.append(dict(dict(model=model, simulation_id=r['simulation_id'], nll=r['nll'],
                                           bounds_reached=';'.join(r.get('bounds_reached') or [])),
                                      **{key:r['params'][key] for key in RECOVERY_PARAMETERS[model]}))
            else:
                excluded.append(dict(model=model, simulation_id=r['simulation_id'], status=r['status'],
                                     reason=r.get('error') or 'Independent local convergence check failed',
                                     projected_gradient_inf=diagnostic.get('projected_gradient_inf'),
                                     probe_improvement_nll=diagnostic.get('best_probe_improvement_nll')))
            for start in (r.get('start_audit') or {}).get('starts', []):
                for key in RECOVERY_PARAMETERS[model]:
                    start_rows.append(dict(model=model, simulation_id=r['simulation_id'], retained=keep,
                                           start_index=start['start_index'], parameter=key,
                                           true_value=recovery_truths()[model][key],
                                           start_value=start['physical_parameters'][key],
                                           relative_error=start['relative_errors'][key]))
            if baseline:
                file = baseline/'raw_fits'/('{}_{:04d}.json'.format(model, r['simulation_id']))
                if not file.is_file():
                    raise FileNotFoundError('Missing paired baseline checkpoint: '+str(file))
                old_record = json.loads(file.read_text(encoding='utf-8'))
                if old_record['observations_sha256'] != r['observations_sha256']:
                    raise ValueError('Baseline observations differ for '+file.name)
                paired.append(dict(model=model, simulation_id=r['simulation_id'], same_data=True,
                                   current_retained=keep, baseline_retained=retained_fit(old_record),
                                   nll_change=r['nll']-old_record['nll'] if keep and retained_fit(old_record) else None))
    summary = pd.DataFrame(summary_rows)
    counts = pd.DataFrame(count_rows)
    summary.to_csv(out/'supplementary_parameter_summary.csv', index=False, float_format='%.12g')
    counts.to_csv(out/'simulation_status_counts.csv', index=False)
    pd.DataFrame(estimates).to_csv(out/'successful_estimates.csv', index=False, float_format='%.12g')
    pd.DataFrame(excluded, columns=['model', 'simulation_id', 'status', 'reason',
                                  'projected_gradient_inf', 'probe_improvement_nll']).to_csv(out/'excluded_runs.csv', index=False)
    starts_frame = pd.DataFrame(start_rows)
    starts_frame.to_csv(out/'initialization_audit.csv', index=False, float_format='%.12g')
    expected_starts = sum(3*len(RECOVERY_PARAMETERS[r['model']]) for r in records if r.get('start_audit'))
    if len(starts_frame) != expected_starts:
        raise AssertionError('Incomplete audit for a successfully initialized trial.')
    deviations = np.abs(starts_frame['start_value']/starts_frame['true_value']-1)
    assert np.all((deviations >= .1-1e-12) & (deviations <= .2+1e-12))
    if paired:
        pd.DataFrame(paired).to_csv(out/'paired_optimizer_comparison.csv', index=False, float_format='%.12g')
    audit = dict(attempted=len(records), retained=sum(retained_fit(r) for r in records),
                 unique_dataset_hashes=len({r['observations_sha256'] for r in records}),
                 paired_hashes_verified=len(paired), physical_start_values_checked=len(starts_frame),
                 smallest_absolute_relative_perturbation=float(deviations.min()),
                 largest_absolute_relative_perturbation=float(deviations.max()),
                 initialization_failed_trials=sum(r.get('start_audit') is None for r in records),
                 total_proposed_start_vectors=sum((r.get('start_audit') or {}).get('total_draws', 0) for r in records),
                 total_rejected_start_vectors=sum((r.get('start_audit') or {}).get('rejected_draws', 0) for r in records),
                 all_datasets_have_one_realization=all(r['R'] == 1 for r in records),
                 successful_only_figures_and_parameter_tables=True)
    (out/'verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (out/'simulation_records.json').write_text(json.dumps(_json_safe(records), ensure_ascii=False), encoding='utf-8')

    note = ('Each model uses {n} independent datasets, one joint bivariate realization on 64 sites per dataset, '
            'with three valid random starts. Every freely estimated physical parameter starts 10–20% above '
            'or below its true value; invalid vectors are rejected and redrawn. The optimizer retains broad '
            'bounds and may leave this starting window. These are truth-informed initialization results. '
            'Only finite, spectrally valid fits passing independent local stationarity checks enter the '
            'boxplots and parameter summaries; counts disclose the exclusions. Local checks do not establish '
            'global optimality. Successful fits at bounds and all their outliers remain included.').format(n=protocol['settings']['simulations'])
    formula_note = ('Bias = mean(estimate − truth); RMSE = sqrt(mean((estimate − truth)^2)); '
                    'IQR = Q3 − Q1. Quantiles use linear interpolation. SD uses n − 1 and MCSE of bias '
                    'is SD/sqrt(n). All are conditional on the retained fits. Boundary fraction indicates '
                    'an active corresponding search bound, including transformed cross-parameter gap constraints. '
                    'Theta is measured in radians; it and rho can be weakly identified when the cross signal is small.')
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Truth-informed simulation results</title>',
            '<style>body{max-width:1200px;margin:30px auto;padding:0 22px;font:15px/1.55 system-ui;color:#213442}'
            'table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{padding:7px 9px;border-bottom:1px solid #d5e0e7;text-align:right}'
            'th{background:#eef4f7}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}a{color:#22688e}.scroll{overflow:auto}</style>',
            '<h1>Figures 4–7: truth-informed initialization</h1>', '<p>'+html.escape(note)+'</p>',
            '<p><a href="supplementary_parameter_summary.csv">Full parameter table (CSV)</a> · '
            '<a href="supplementary_tables.tex">LaTeX tables</a> · <a href="successful_estimates.csv">Retained estimates</a> · '
            '<a href="excluded_runs.csv">Excluded run log</a> · <a href="truth_started_boxplot_experiments.py">Python script</a></p>',
            counts.to_html(index=False, border=0), '<p>'+html.escape(formula_note)+'</p>']
    tex = [r'\documentclass[10pt]{article}', r'\usepackage[margin=1.4cm]{geometry}',
           r'\usepackage{booktabs,longtable,amsmath}', r'\begin{document}',
           r'\section*{Parameter recovery with truth-informed initialization}',
           note.replace('%', r'\%').replace('–', '--'), r'\par\medskip',
           r'For retained estimates $\widehat\eta_1,\ldots,\widehat\eta_n$ and truth $\eta_0$, '
           r'$\mathrm{Bias}=n^{-1}\sum_i(\widehat\eta_i-\eta_0)$ and '
           r'$\mathrm{RMSE}=\sqrt{n^{-1}\sum_i(\widehat\eta_i-\eta_0)^2}$. '
           r'$\mathrm{IQR}=Q_3-Q_1$; quantiles use linear interpolation. '
           r'SD uses denominator $n-1$, and the Monte Carlo standard error of bias is $\mathrm{SD}/\sqrt{n}$. '
           r'All summaries condition on retained fits. Bound fraction includes corresponding transformed '
           r'cross-parameter gap constraints. The angle $\theta$ is in radians.',
           r'\par\medskip']
    columns = ['parameter', 'true_value', 'mean', 'bias', 'RMSE', 'median', 'Q1', 'Q3', 'IQR', 'boundary_hit_fraction']
    for model in RECOVERY_MODELS:
        table = summary[summary.model == model].copy()
        count = counts[counts.model == model].iloc[0]
        title = '{}: {} retained / {} attempted ({} excluded)'.format(RECOVERY_NAMES[model], count.retained, count.attempted, count.excluded)
        base = 'figure_{:02d}_{}_boxplots'.format(RECOVERY_FIGURES[model], model)
        page.extend(['<h2>'+html.escape(title)+'</h2>', '<div class="scroll">'+table[columns].to_html(
            index=False, border=0, float_format=lambda x:'{:.4g}'.format(x))+'</div>',
                     '<a href="'+base+'.pdf"><img src="'+base+'.png" alt="'+html.escape(title)+'"></a>'])
        if model != RECOVERY_MODELS[0]:
            tex.append(r'\clearpage')
        tex.append(r'\subsection*{'+title.replace('Matérn', r'Mat\'ern').replace('_', r'\_')+'}')
        tex.append(r'\begin{center}\small\begin{tabular}{lrrrrrrrrr}\toprule')
        tex.append(r'Parameter & True & Mean & Bias & RMSE & Median & Q1 & Q3 & IQR & Bound frac. \\ \midrule')
        for _, row in table.iterrows():
            label = RECOVERY_LABELS[row.parameter].replace(' (radians)', '')
            nums = ['{:.4g}'.format(row[key]) for key in columns[1:]]
            tex.append(label+' & '+' & '.join(nums)+r' \\')
        tex.append(r'\bottomrule\end{tabular}\end{center}')
    page.append('</html>')
    tex.append(r'\end{document}')
    (out/'supplementary_tables.html').write_text('\n'.join(page), encoding='utf-8')
    (out/'index.html').write_text('\n'.join(page), encoding='utf-8')
    (out/'supplementary_tables.tex').write_text('\n'.join(tex), encoding='utf-8')
    readme = ['# Truth-informed parameter recovery', note, counts.to_string(index=False), formula_note,
              'Only starting values are close to truth. This protocol cannot be used unchanged for real data, '
              'where the true parameters are unknown. Excluding failures changes the target to recovery conditional '
              'on numerical success. The initial sign/magnitude draws are independent; conditioning on joint '
              'spectral validity induces dependence between accepted starting parameters.',
              'Generating parameters: '+json.dumps(recovery_truths(), indent=2),
              'p=1, direction=(1,0), means=0, marginal sills=1, nugget=0. '
              'Locations are the same 8 by 8 grid on [0,10]^2. Covariances use the corrected Fourier formulas. '
              'Spectral validity uses continuous-frequency component maxima and a sufficient mixed bound. '
              'There is no covariance eigenvalue clipping or added nugget.',
              'Full fitted search bounds, seeds, numerical tolerances and software are in experiment_protocol.json. '
              'The CSV summary additionally includes SD and MCSE of bias. initialization_audit.csv records every '
              'accepted starting value. raw_fits/ and simulation_records.json retain all attempts for auditing only; '
              'failed runs contribute no estimates to successful_estimates.csv, boxplots or parameter summary tables.',
              '## Reproduce\n```bash\npython truth_started_boxplot_experiments.py --output results --workers 8\n```\n'
              'Defaults: 200 datasets per model, exactly one realization per dataset, three starts. '
              'Add --resume to continue checkpoints. Use --stage summaries to rebuild figures/tables. '
              'Optional --baseline PATH verifies paired observations against the preceding experiment.',
              'The four PNG/PDF files replace Figures 4–7 for this initialization protocol. Figures 1–3 are unaffected.']
    (out/'README.md').write_text('\n\n'.join(readme)+'\n', encoding='utf-8')
    source = Path(__file__).resolve()
    if source != out/source.name:
        shutil.copy2(source, out/source.name)
    (out/'requirements.txt').write_text('numpy>=1.22\nscipy>=1.10\npandas>=2.0\nmatplotlib>=3.6\nthreadpoolctl\n', encoding='utf-8')
    with zipfile.ZipFile(out.parent/(out.name+'.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(out.rglob('*')):
            if file.is_file():
                archive.write(file, str(Path(out.name)/file.relative_to(out)))
    print(counts.to_string(index=False), flush=True)
    print('Verification:', json.dumps(audit), flush=True)
    return summary, counts


def main_truth_experiments():
    parser = argparse.ArgumentParser(description='Figures 4-7 with random physical starts 10-20% from truth.')
    parser.add_argument('--output', type=Path, default=Path('boxplots_truth_starts_10_20_percent'))
    parser.add_argument('--simulations', type=int, default=200)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--maxiter', type=int, default=300)
    parser.add_argument('--stage', choices=['all', 'summaries'], default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if min(args.simulations, args.workers, args.maxiter) < 1:
        parser.error('Counts must be positive.')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(simulations=args.simulations, workers=args.workers, seed=args.seed,
                    maxiter=args.maxiter, starts=3, candidates=3, replicates=1)
    _, prepared = prepare_recovery(settings)
    protocol = dict(settings=settings, initialization='physical truth*(1+sign*Uniform(.10,.20)); '
                    'whole-vector rejection for validity; three valid starts per dataset; no exact-truth starts',
                    initialization_is_truth_informed=True, truth=recovery_truths(), p=1, N=64, R=1,
                    geometry='8 by 8 equally spaced on [0,10]^2', direction=[1, 0],
                    means=[0, 0], marginal_sills=[1, 1], nugget=0,
                    optimizer_policy='Optimize all three supplied starts, then adaptive local polishing as needed; '
                    'no generic/random-global or reflected-phase extra starts; original broad fitted bounds.',
                    success_policy='Finite spectrally valid estimate passing independent projected-gradient and finite-scale local probes.',
                    fit_configurations={name:asdict(recovery_config(name, settings, args.seed)) for name in RECOVERY_MODELS},
                    local_check_options=asdict(RobustOptions()),
                    generating_validity={name:dict(minimum_eigenvalue=item['minimum_eigenvalue'],
                        condition_number=item['condition_number'], spectral_ratio_bound=item['params']['spectral_ratio_bound'])
                        for name, item in prepared.items()},
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    pandas=pd.__version__, matplotlib=matplotlib.__version__,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol_file = out/'experiment_protocol.json'
    if protocol_file.exists():
        previous = json.loads(protocol_file.read_text(encoding='utf-8'))
        if previous['settings'] != settings:
            raise ValueError('Existing directory has different settings; choose a new directory.')
        if args.stage == 'all' and not args.resume:
            raise FileExistsError('Use --resume or a new output directory.')
        if previous['script_sha256'] != protocol['script_sha256']:
            raise ValueError('Source changed since checkpoints were made; use a new directory.')
    protocol_file.write_text(json.dumps(_json_safe(protocol), indent=2), encoding='utf-8')
    raw = out/'raw_fits'
    raw.mkdir(exist_ok=True)
    records = [json.loads(file.read_text(encoding='utf-8')) for file in sorted(raw.glob('*.json'))]
    print('Output:', out, flush=True)
    if args.stage == 'all':
        completed = {(r['model'], r['simulation_id']) for r in records}
        tasks = [(name, i) for i in range(args.simulations) for name in RECOVERY_MODELS if (name, i) not in completed]
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_experiment_worker,
                                 initargs=(settings,)) as executor:
            futures = {executor.submit(run_truth_task, task):task for task in tasks}
            for future in as_completed(futures):
                name, simulation_id = futures[future]
                result = future.result()
                file = raw/('{}_{:04d}.json'.format(name, simulation_id))
                file.write_text(json.dumps(_json_safe(result), indent=2), encoding='utf-8')
                records.append(result)
                if len(records)%8 == 0 or len(records) == args.simulations*4:
                    print('{}/{} fits; {} retained; {:.1f} minutes'.format(len(records), args.simulations*4,
                          sum(retained_fit(r) for r in records), (time.perf_counter()-started)/60), flush=True)
    write_success_reports(out, records, protocol, args.baseline.resolve() if args.baseline else None)


if __name__ == '__main__':
    main_truth_experiments()
