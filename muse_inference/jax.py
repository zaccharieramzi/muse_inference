
from datetime import datetime, timedelta
from functools import partial

from numpy.random import SeedSequence

import jax
from jax import jacfwd, vmap, grad, jvp
from jax.flatten_util import ravel_pytree
from jax.numpy import array, atleast_1d, atleast_2d, concatenate, mean
from jax.numpy.linalg import inv
from jax.scipy.optimize import minimize
from jax.scipy.sparse.linalg import cg

from .muse_inference import MuseProblem, MuseResult, ScoreAndMAP


class JaxMuseProblem(MuseProblem):

    def sample_x_z(self, θ):
        raise NotImplementedError()

    def logLike(self, x, z, θ):
        raise NotImplementedError()

    def logPrior(self, θ):
        raise NotImplementedError()


    def __init__(self, implicit_diff=True, jit=True):

        super().__init__()

        self.np = jax.numpy

        if implicit_diff:
            
            def _get_H_i(self, rng, *, θ, method=None, θ_tol=None, z_tol=None, step=None, skip_errors=False):

                try:

                    (x, z) = self.sample_x_z(rng, θ)
                    z_MAP_guess = self.z_MAP_guess_from_truth(x, z, θ)
                    z_MAP = self.z_MAP_and_score(x, z_MAP_guess, θ, method=method, θ_tol=θ_tol, z_tol=z_tol).z

                    θ_raveled, z_MAP_raveled = self.ravel_θ(θ), self.ravel_z(z_MAP)
                    unravel_θ, unravel_z = self.unravel_θ, self.unravel_z

                    # non-implicit-diff term
                    H1 = jacfwd(lambda θ1: grad(lambda θ2: self.logLike(self.sample_x_z(rng, unravel_θ(θ1))[0], z_MAP, unravel_θ(θ2)))(θ_raveled))(θ_raveled)

                    # term involving dzMAP/dθ via implicit-diff (w/ conjugate-gradient linear solve)
                    cg_kwargs = dict(tol=z_tol) if z_tol is not None else dict()
                    dFdθ = jacfwd(lambda θ: grad(lambda z: self.logLike(x, unravel_z(z), unravel_θ(θ)))(z_MAP_raveled))(θ_raveled)
                    dFdθ1 = jacfwd(lambda θ1: grad(lambda z: self.logLike(self.sample_x_z(rng, unravel_θ(θ1))[0], unravel_z(z), θ))(z_MAP_raveled))(θ_raveled)
                    inv_dFdz_dFdθ1 = jax.vmap(lambda vec: cg(lambda vec: jvp(lambda z: grad(lambda z: self.logLike(x, unravel_z(z), θ))(z), (z_MAP_raveled,), (vec,))[1], vec, **cg_kwargs)[0], in_axes=1, out_axes=1)(dFdθ1)
                    H2 = -dFdθ.T @ inv_dFdz_dFdθ1

                    return H1 + H2

                except Exception:
                    if skip_errors:
                        return None
                    else:
                        raise

            self._get_H_i = _get_H_i.__get__(self)
            if jit:
                self._get_H_i = jax.jit(self._get_H_i, static_argnames=("self", "method", "skip_errors"))

        if jit:
            self.sample_x_z = jax.jit(self.sample_x_z, static_argnames=("self",))
            self.logLike = jax.jit(self.logLike, static_argnames=("self",))
            self.logPrior = jax.jit(self.logPrior, static_argnames=("self",))
            self.val_gradz_gradθ_logLike = jax.jit(self.val_gradz_gradθ_logLike, static_argnames=("self",))
            self.gradθ_hessθ_logPrior = jax.jit(self.gradθ_hessθ_logPrior, static_argnames=("self",))
            self.z_MAP_and_score = jax.jit(self.z_MAP_and_score, static_argnames=("self", "method"))

    def val_gradz_gradθ_logLike(self, x, z, θ, transformed_θ=None):
        logLike, (gradz_logLike, gradθ_logLike) = jax.value_and_grad(self.logLike, argnums=(1, 2))(x, z, θ)
        return (logLike, gradz_logLike, gradθ_logLike)

    def z_MAP_and_score(self, x, z_guess, θ, method=None, options=dict(), z_tol=None, θ_tol=None):

        if z_tol is not None:
            options = dict(gtol=z_tol, **options)
        if method is None:
            method = "l-bfgs-experimental-do-not-rely-on-this"

        ravel, unravel = self._ravel_unravel(z_guess)
        
        soln = minimize(
            lambda z_vec: -self.logLike(x, unravel(z_vec), θ), 
            ravel(z_guess), 
            method = method,
            options = options
        )

        zMAP = unravel(soln.x)

        gradθ = self.val_gradz_gradθ_logLike(x, zMAP, θ)[2]

        return ScoreAndMAP(gradθ, gradθ, zMAP, soln)

    def gradθ_hessθ_logPrior(self, θ, transformed_θ=None):
        g = grad(self.logPrior)(θ)
        H = jax.hessian(self.logPrior)(θ)
        return (g, H)

    def _ravel_unravel(self, x):
        ravel = lambda x_tree: ravel_pytree(x_tree)[0]
        unravel = ravel_pytree(x)[1]
        return (ravel, unravel)

    def _split_rng(self, key, N):
        return jax.random.split(key, N)

    def _default_rng(self):
        return jax.random.PRNGKey(SeedSequence().generate_state(1)[0])
