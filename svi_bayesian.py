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
        f_nk     ~ Normal(0, 1)              (N, K)   local latent
        tau      ~ Gamma(1,1)               scalar
        X_ni     ~ Normal(f @ W.T, 1/√tau)  (N, P)

    WHY NO MINI-BATCHING
    --------------------
    f is a *local* latent variable — one K-vector per observation row.
    AutoDiagonalNormal builds one variational parameter per latent dim,
    so it needs to know N at guide-construction time and hold N*K
    variational params.  If we pass a mini-batch of size B < N, the plate
    declares size=N but the guide only ever sees B rows, producing:
        "Shape mismatch inside plate('time_steps') at site obs dim -1, N vs B"
    The correct fix is to pass the FULL window matrix X to svi.step()
    every iteration (no mini-batching).  Window sizes are at most
    4032 × 25 = ~100k floats — trivially fits in memory.
    """
    n_obs    = X.shape[0]
    n_assets = X.shape[1]

    # ── ARD precisions ────────────────────────────────────────────────────────
    alpha = pyro.sample(
        "alpha",
        dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1)
    )
    alpha_k = alpha.reshape(-1)                         # (K,) — strips Predictive's leading dims

    # ── Factor loadings W: (P, K) ─────────────────────────────────────────────
    W_scale = (1.0 / torch.sqrt(alpha_k)).unsqueeze(0).expand(n_assets, n_factors)
    W = pyro.sample(
        "W",
        dist.Normal(torch.zeros(n_assets, n_factors), W_scale).to_event(2)
    )

    # ── Observation noise ─────────────────────────────────────────────────────
    tau   = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)

    # ── Factor scores f + likelihood — full N rows, no mini-batching ──────────
    with pyro.plate("time_steps", n_obs):
        f = pyro.sample(
            "f",
            dist.Normal(torch.zeros(n_factors), torch.ones(n_factors)).to_event(1)
        )                                               # (N, K)

        # .transpose(-2,-1) works for both 2-D (N,K)→(K,N) and 3-D from Predictive
        # torch.matmul broadcasts over any leading sample dims added by Predictive
        mean = torch.matmul(f, W.transpose(-2, -1))    # (N, P)

        pyro.sample(
            "obs",
            dist.Normal(mean, sigma).to_event(1),
            obs=X
        )


def train_svi_model(X, n_factors=5, lr=0.01, iterations=500, batch_size=None):
    """
    Train via SVI and return per-asset loading magnitude scores.

    Args:
        X:          np.ndarray (N, P) — standardised returns
        n_factors:  K
        lr:         Adam learning rate
        iterations: SVI gradient steps
        batch_size: ignored — kept for API compatibility.
                    Full X is always used per step (see model docstring).

    Returns:
        scores: np.ndarray (P,) — sum |posterior mean W| per asset
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)

    pyro.clear_param_store()

    guide     = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": lr})
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Pass full X every step — no mini-batching of local latent f
    for step in range(iterations):
        loss = svi.step(X_tensor, n_factors)
        if step % 100 == 0:
            print(f"    Step {step:4d}: ELBO loss = {loss:.4f}")

    # Posterior W samples using full data
    predictive = Predictive(model, guide=guide, num_samples=200,
                            return_sites=["W"])
    with torch.no_grad():
        posterior = predictive(X_tensor, n_factors)

    # W_samples: (200, P, K) → mean (P, K) → scores (P,)
    W_mean = posterior["W"].mean(dim=0).numpy()
    scores = np.sum(np.abs(W_mean), axis=1)             # (P,)
    return scores
