import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal
import numpy as np

pyro.enable_validation(True)


def model(X, n_factors):
    """
    Probabilistic factor model with ARD priors.

    Generative process:
        alpha_k  ~ Gamma(1, 1)                     k=1..K   (ARD precisions)
        W_ik     ~ Normal(0, 1/sqrt(alpha_k))       i=1..P, k=1..K  (loadings)
        f_nk     ~ Normal(0, 1)                     n=1..N, k=1..K  (scores)
        tau      ~ Gamma(1, 1)                                        (noise precision)
        X_ni     ~ Normal( sum_k f_nk * W_ik, 1/sqrt(tau) )

    Shape conventions used throughout:
        N = n_obs  (rows / time-steps in the window)
        P = n_assets (columns / ETFs)
        K = n_factors

    All bugs fixed vs original:
    ---------------------------
    BUG 1: `dist.Normal(0, 1).to_event(1)` on a scalar distribution has
            batch_shape=() so to_event(1) raises
            "reinterpreted_batch_ndims <= len(batch_shape), actual 1 vs 0".
            Fix: expand the distribution to the correct shape FIRST, then
            use to_event to mark the last dimension as an event dimension.

    BUG 2: Nested `pyro.plate("factors", ...)` inside `pyro.plate("assets", ...)`
            produces W with shape (n_factors, n_assets) — transposed relative to
            what the matmul `f @ W.T` expects, and also confuses AutoDiagonalNormal
            when it tries to infer the parameter shapes from the trace.
            Fix: sample W as a single (n_assets, n_factors) tensor using
            .expand([n_assets, n_factors]).to_event(2) inside one plate.

    BUG 3: Nested `pyro.plate("factors", ...)` inside `pyro.plate("observations", ...)`
            for factor scores f produces shape (n_factors, n_obs) — also transposed —
            and the inner plate name "factors" collides with the one used for W.
            Fix: sample f as a single (n_obs, n_factors) tensor inside one
            "observations" plate using .expand([n_obs, n_factors]).to_event(1).

    BUG 4: `mean = torch.mm(f, W.t())` assumes f:(n_obs,K) and W:(n_assets,K)
            which is only correct if the shapes above are right.  After fixing
            BUG 2 & 3 this matmul is correct; left as-is but now actually works.

    BUG 5: The outer `pyro.plate("data", n_obs)` around the likelihood conflicts
            with the "observations" plate used to sample f — Pyro sees two
            independent plate contexts with the same batch dimension, causing
            ELBO mis-accounting.  Fix: use a single "obs_plate" and remove the
            duplicate outer plate.
    """
    n_obs    = X.shape[0]
    n_assets = X.shape[1]

    # ── ARD precisions: one per factor ──────────────────────────────────────
    # Shape: (K,)  →  event_shape: (K,)
    alpha = pyro.sample(
        "alpha",
        dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1)
    )                                                  # (K,)

    # ── Factor loadings W: (P, K) ────────────────────────────────────────────
    # BUG 1+2 FIX: expand to (n_assets, n_factors) then mark both dims as event.
    # alpha is (K,) so 1/sqrt(alpha) broadcasts correctly across the P dimension.
    W = pyro.sample(
        "W",
        dist.Normal(
            torch.zeros(n_assets, n_factors),
            (1.0 / torch.sqrt(alpha)).unsqueeze(0).expand(n_assets, n_factors)
        ).to_event(2)
    )                                                  # (P, K)

    # ── Factor scores f: (N, K) ───────────────────────────────────────────────
    # BUG 3 FIX: single plate over observations, f sampled as (N, K) event tensor.
    with pyro.plate("obs_plate", n_obs):
        f = pyro.sample(
            "f",
            dist.Normal(
                torch.zeros(n_factors),
                torch.ones(n_factors)
            ).to_event(1)
        )                                              # (N, K)

    # ── Observation noise ─────────────────────────────────────────────────────
    tau   = pyro.sample("tau", dist.Gamma(1.0, 1.0))  # scalar
    sigma = 1.0 / torch.sqrt(tau)

    # ── Likelihood ────────────────────────────────────────────────────────────
    # f: (N, K),  W: (P, K)  →  mean: (N, P)
    mean = torch.mm(f, W.t())                         # (N, P)

    # BUG 5 FIX: use the same plate "obs_plate" so Pyro knows N is already
    # accounted for.  obs has shape (N, P) with event_dim=1 (the P assets).
    with pyro.plate("obs_plate", n_obs):
        pyro.sample(
            "obs",
            dist.Normal(mean, sigma).to_event(1),
            obs=X
        )


def train_svi_model(X, n_factors=5, lr=0.01, iterations=500, batch_size=32):
    """
    Train the Bayesian factor model via SVI and return per-asset loading scores.

    Args:
        X:          np.ndarray of shape (N, P) — standardised returns matrix
        n_factors:  number of latent factors K
        lr:         Adam learning rate
        iterations: number of SVI gradient steps
        batch_size: mini-batch size (rows sampled per step)

    Returns:
        scores: np.ndarray of shape (P,) — sum of |posterior mean loadings|
                per asset; higher = stronger systematic exposure
    """
    n_obs    = X.shape[0]
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Clear any cached param store from previous calls (important in loops)
    pyro.clear_param_store()

    guide     = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": lr})
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(iterations):
        # Mini-batch over observations (rows)
        idx   = np.random.choice(n_obs, min(batch_size, n_obs), replace=False)
        batch = X_tensor[idx]                          # (batch_size, P)
        loss  = svi.step(batch, n_factors)
        if step % 100 == 0:
            print(f"    Step {step:4d}: ELBO loss = {loss:.4f}")

    # ── Posterior inference ───────────────────────────────────────────────────
    # Use full data for posterior predictive (no mini-batching at inference time)
    predictive = Predictive(model, guide=guide, num_samples=200,
                            return_sites=["W"])
    with torch.no_grad():
        posterior = predictive(X_tensor, n_factors)

    # W posterior samples: (num_samples, P, K)
    W_samples = posterior["W"]                         # (200, P, K)
    W_mean    = W_samples.mean(dim=0).numpy()          # (P, K)

    # Score = total absolute loading magnitude per asset
    scores = np.sum(np.abs(W_mean), axis=1)            # (P,)
    return scores
