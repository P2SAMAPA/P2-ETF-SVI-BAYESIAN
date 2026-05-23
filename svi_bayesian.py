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

    Shapes:  N = n_obs, P = n_assets, K = n_factors

        alpha_k  ~ Gamma(1,1)               (K,)
        W_ik     ~ Normal(0, 1/sqrt(alpha))  (P, K)
        f_nk     ~ Normal(0, 1)              (N, K)
        tau      ~ Gamma(1,1)               scalar
        X_ni     ~ Normal(f @ W.T, 1/√tau)  (N, P)

    FIX: AutoDiagonalNormal injects an extra leading sample dimension when
    it calls the model during Predictive.  alpha arrives as shape (..., K)
    where ... may be (1,) or (num_samples,).  We must squeeze/reshape alpha
    to exactly (K,) before broadcasting it into the W prior scale, otherwise
    `.expand([n_assets, n_factors])` receives a 3-D tensor with only 2
    target dims and raises:
        "expand(FloatTensor{[1,1,5]}, size=[7,5]): number of sizes provided
         (2) must be >= number of dimensions in the tensor (3)"
    """
    n_obs    = X.shape[0]
    n_assets = X.shape[1]

    # ── ARD precisions ────────────────────────────────────────────────────────
    alpha = pyro.sample(
        "alpha",
        dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1)
    )                                           # declared shape: (K,)
                                                # but Predictive may pass (1,K)

    # FIX: flatten to (K,) regardless of any leading sample/batch dims added
    # by AutoDiagonalNormal during the Predictive posterior sweep.
    alpha_k = alpha.reshape(-1)                 # always (K,)

    # ── Factor loadings W: (P, K) ─────────────────────────────────────────────
    # Build scale as (P, K) cleanly from the (K,) alpha_k.
    W_scale = (1.0 / torch.sqrt(alpha_k)).unsqueeze(0).expand(n_assets, n_factors)
    W = pyro.sample(
        "W",
        dist.Normal(torch.zeros(n_assets, n_factors), W_scale).to_event(2)
    )                                           # (P, K)

    # ── Observation noise ─────────────────────────────────────────────────────
    tau   = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)

    # ── Factor scores + likelihood — plate name must not match any sample name ─
    with pyro.plate("time_steps", n_obs):
        f = pyro.sample(
            "f",
            dist.Normal(torch.zeros(n_factors), torch.ones(n_factors)).to_event(1)
        )                                       # (N, K)

        mean = torch.mm(f, W.t())              # (N, P)

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
        iterations: SVI steps
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

    # ── Training ──────────────────────────────────────────────────────────────
    for step in range(iterations):
        idx   = np.random.choice(n_obs, min(batch_size, n_obs), replace=False)
        batch = X_tensor[idx]
        loss  = svi.step(batch, n_factors)
        if step % 100 == 0:
            print(f"    Step {step:4d}: ELBO loss = {loss:.4f}")

    # ── Posterior W via Predictive ────────────────────────────────────────────
    predictive = Predictive(model, guide=guide, num_samples=200,
                            return_sites=["W"])
    with torch.no_grad():
        posterior = predictive(X_tensor, n_factors)

    # W_samples: (num_samples, P, K)  — mean over samples → (P, K)
    W_mean = posterior["W"].mean(dim=0).numpy()
    scores = np.sum(np.abs(W_mean), axis=1)     # (P,)
    return scores
