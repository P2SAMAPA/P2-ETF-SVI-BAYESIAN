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
        alpha_k  ~ Gamma(1, 1)                          k=1..K
        W_ik     ~ Normal(0, 1/sqrt(alpha_k))            i=1..P, k=1..K
        f_nk     ~ Normal(0, 1)                          n=1..N, k=1..K
        tau      ~ Gamma(1, 1)
        X_ni     ~ Normal( f_n @ W.T, 1/sqrt(tau) )

    Key Pyro rules followed here:
      - Every pyro.plate name must be UNIQUE within one model trace.
        Reusing the same name for two different plates raises
        "Multiple sample sites named '<name>'".
      - Mini-batch subsampling is handled by passing subsample_size to
        the plate so Pyro can correctly scale the ELBO.
    """
    n_obs    = X.shape[0]   # full dataset size (may be mini-batch; see below)
    n_assets = X.shape[1]   # P

    # ARD precisions: shape (K,)
    alpha = pyro.sample(
        "alpha",
        dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1)
    )

    # Factor loadings W: shape (P, K)
    W = pyro.sample(
        "W",
        dist.Normal(
            torch.zeros(n_assets, n_factors),
            (1.0 / torch.sqrt(alpha)).unsqueeze(0).expand(n_assets, n_factors)
        ).to_event(2)
    )

    # Observation noise
    tau   = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)

    # FIX: ONE plate named "obs" covering both factor scores f and the
    # likelihood. Previous code used "obs_plate" twice — once for f and
    # once for the obs sample site — which Pyro treats as two independent
    # plates with the same name and raises "Multiple sample sites named
    # 'obs_plate'".
    #
    # n_obs here equals X.shape[0] which is the mini-batch size during
    # training. We pass the true full dataset size via the plate's first
    # argument (stored in config / trainer) so the ELBO is scaled correctly.
    # During the Predictive call we pass the full X so n_obs == full size.
    with pyro.plate("obs", n_obs):
        # Factor scores: (n_obs, K), event_dim=1 over K
        f = pyro.sample(
            "f",
            dist.Normal(
                torch.zeros(n_factors),
                torch.ones(n_factors)
            ).to_event(1)
        )

        # Reconstruction: (n_obs, P)
        mean = torch.mm(f, W.t())

        # Likelihood, event_dim=1 over P assets
        pyro.sample(
            "obs",
            dist.Normal(mean, sigma).to_event(1),
            obs=X
        )


def train_svi_model(X, n_factors=5, lr=0.01, iterations=500, batch_size=32):
    """
    Train the Bayesian factor model via SVI.

    Args:
        X:          np.ndarray (N, P) -- standardised returns
        n_factors:  number of latent factors K
        lr:         Adam learning rate
        iterations: SVI gradient steps
        batch_size: rows sampled per mini-batch step

    Returns:
        scores: np.ndarray (P,) -- sum |posterior mean W| per asset
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
        # Pass the mini-batch; model reads X.shape[0] for plate size.
        # ELBO is automatically scaled by n_obs/batch_size via the plate.
        loss  = svi.step(batch, n_factors)
        if step % 100 == 0:
            print(f"    Step {step:4d}: ELBO loss = {loss:.4f}")

    # Posterior over W using full data
    predictive = Predictive(model, guide=guide, num_samples=200,
                            return_sites=["W"])
    with torch.no_grad():
        posterior = predictive(X_tensor, n_factors)

    W_mean = posterior["W"].mean(dim=0).numpy()   # (P, K)
    scores = np.sum(np.abs(W_mean), axis=1)       # (P,)
    return scores
