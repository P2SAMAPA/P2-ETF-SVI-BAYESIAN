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

    Shapes:  N=n_obs, P=n_assets, K=n_factors

        alpha_k  ~ Gamma(1,1)               (K,)
        W_ik     ~ Normal(0, 1/sqrt(alpha))  (P, K)
        f_nk     ~ Normal(0, 1)              (N, K)
        tau      ~ Gamma(1,1)               scalar
        X_ni     ~ Normal(f @ W.T, 1/√tau)  (N, P)

    Pyro's Predictive adds a leading sample dimension to every latent
    variable.  During posterior sampling:
        alpha → (..., K)
        W     → (..., P, K)   ← 3-D, so .t() crashes
        tau   → (...,)

    Fixes applied
    -------------
    1. alpha.reshape(-1) → always (K,) before building W_scale.
    2. W.transpose(-2, -1) instead of W.t() → works for any number of
       leading dims (2-D or 3-D), transposes last two axes only.
    3. torch.matmul(f, W.transpose(-2,-1)) instead of torch.mm → mm
       requires exactly 2-D inputs; matmul broadcasts over batch dims.
    """
    n_obs    = X.shape[0]
    n_assets = X.shape[1]

    # ── ARD precisions ────────────────────────────────────────────────────────
    alpha = pyro.sample(
        "alpha",
        dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1)
    )
    # Flatten to (K,) regardless of leading sample/batch dims from Predictive
    alpha_k = alpha.reshape(-1)                     # (K,)

    # ── Factor loadings W: (P, K) ─────────────────────────────────────────────
    W_scale = (1.0 / torch.sqrt(alpha_k)).unsqueeze(0).expand(n_assets, n_factors)
    W = pyro.sample(
        "W",
        dist.Normal(torch.zeros(n_assets, n_factors), W_scale).to_event(2)
    )                                               # declared (P, K)
                                                    # Predictive adds (..., P, K)

    # ── Observation noise ─────────────────────────────────────────────────────
    tau   = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)

    # ── Factor scores + likelihood ────────────────────────────────────────────
    with pyro.plate("time_steps", n_obs):
        f = pyro.sample(
            "f",
            dist.Normal(torch.zeros(n_factors), torch.ones(n_factors)).to_event(1)
        )                                           # (N, K)

        # FIX: use .transpose(-2, -1) instead of .t() so it works when
        # Predictive injects W as (..., P, K) with leading sample dims.
        # FIX: use torch.matmul instead of torch.mm — mm requires exactly
        # 2-D inputs; matmul handles (..., N, K) @ (..., K, P) → (..., N, P).
        W_T  = W.transpose(-2, -1)                 # (..., K, P)
        mean = torch.matmul(f, W_T)                # (..., N, P)

        pyro.sample(
            "obs",
            dist.Normal(mean, sigma).to_event(1),
            obs=X
        )


def train_svi_model(X, n_factors=5, lr=0.01, iterations=500, batch_size=32):
    """
    Train via SVI and return per-asset loading magnitude scores.

    Args:
        X:          np.ndarray (N, P) — standardised returns
        n_factors:  K
        lr:         Adam learning rate
        iterations: SVI gradient steps
        batch_size: mini-batch rows per step

    Returns:
        scores: np.ndarray (P,) — sum |posterior mean W| per asset
    """
    n_obs    = X.shape[0]
    X_tensor = torch.tensor(X, dtype=torch.float32)

    pyro.clear_param_store()

    guide     = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": lr})
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for step in range(iterations):
        idx   = np.random.choice(n_obs, min(batch_size, n_obs), replace=False)
        batch = X_tensor[idx]
        loss  = svi.step(batch, n_factors)
        if step % 100 == 0:
            print(f"    Step {step:4d}: ELBO loss = {loss:.4f}")

    # Posterior W samples using full data
    predictive = Predictive(model, guide=guide, num_samples=200,
                            return_sites=["W"])
    with torch.no_grad():
        posterior = predictive(X_tensor, n_factors)

    # W_samples: (num_samples, P, K) → mean (P, K) → scores (P,)
    W_samples = posterior["W"]                      # (200, P, K)
    W_mean    = W_samples.mean(dim=0).numpy()       # (P, K)
    scores    = np.sum(np.abs(W_mean), axis=1)      # (P,)
    return scores
