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

    THE BUG: pyro.plate(name, ...) registers 'name' as a sample site in
    Pyro's trace.  If any pyro.sample() inside or outside the model uses
    the same string as the plate name, Pyro raises:
        "Multiple sample sites named '<name>'"
    The observed variable was named "obs" and the plate was also named "obs"
    — a direct collision.  Fix: rename the plate to "time_steps" (or any
    string that doesn't match any pyro.sample site name in the model).

    Generative process:
        alpha_k  ~ Gamma(1, 1)                     k=1..K
        W_ik     ~ Normal(0, 1/sqrt(alpha_k))       i=1..P, k=1..K
        f_nk     ~ Normal(0, 1)                     n=1..N, k=1..K
        tau      ~ Gamma(1, 1)
        X_ni     ~ Normal( f_n @ W.T, 1/sqrt(tau) )
    """
    n_obs    = X.shape[0]
    n_assets = X.shape[1]   # P

    # ── ARD precisions: (K,) ─────────────────────────────────────────────────
    alpha = pyro.sample(
        "alpha",
        dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1)
    )

    # ── Factor loadings W: (P, K) ─────────────────────────────────────────────
    W = pyro.sample(
        "W",
        dist.Normal(
            torch.zeros(n_assets, n_factors),
            (1.0 / torch.sqrt(alpha)).unsqueeze(0).expand(n_assets, n_factors)
        ).to_event(2)
    )

    # ── Noise precision ───────────────────────────────────────────────────────
    tau   = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)

    # ── Factor scores + likelihood — plate named "time_steps" ─────────────────
    # CRITICAL FIX: plate name must not equal any pyro.sample site name.
    # Previous attempts used "obs_plate" (collided when reused) and "obs"
    # (collided with the pyro.sample("obs", ...) site inside the plate).
    # "time_steps" is unique — no sample site uses this name.
    with pyro.plate("time_steps", n_obs):
        # f: (n_obs, K), event_dim=1
        f = pyro.sample(
            "f",
            dist.Normal(
                torch.zeros(n_factors),
                torch.ones(n_factors)
            ).to_event(1)
        )

        # mean: (n_obs, P)
        mean = torch.mm(f, W.t())

        # observed data: (n_obs, P), event_dim=1
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
        n_factors:  number of latent factors K
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

    W_mean = posterior["W"].mean(dim=0).numpy()   # (P, K)
    scores = np.sum(np.abs(W_mean), axis=1)       # (P,)
    return scores
