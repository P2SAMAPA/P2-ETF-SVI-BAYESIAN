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

    Shapes:  N = X.shape[0] (observations), P = X.shape[1] (assets), K = n_factors

        alpha_k  ~ Gamma(1,1)               (K,)     ARD precisions
        W_ik     ~ Normal(0, 1/sqrt(alpha))  (P, K)   factor loadings
        f_nk     ~ Normal(0, 1)              (N, K)   factor scores (local latent)
        tau      ~ Gamma(1,1)               scalar   noise precision
        X_ni     ~ Normal(f @ W.T, 1/√tau)  (N, P)   observed returns

    All fixes vs the original repo code
    ------------------------------------
    BUG 1  Nested plates for W (assets x factors) produced wrong shape and
           confused AutoDiagonalNormal.  Fix: sample W as a single (P,K)
           tensor with .to_event(2) — no nested plates.

    BUG 2  Nested plates for f (observations x factors) with the name
           "factors" colliding with the W plate.  Fix: sample f as a (K,)
           event inside a single "time_steps" plate over N.

    BUG 3  Plate named "data" for the likelihood conflicted with the
           "observations" plate for f (both over N).  Fix: single
           "time_steps" plate covers both f and obs.

    BUG 4  Plate name "obs" (or "obs_plate") collided with the sample
           site name "obs".  Fix: plate is named "time_steps" — unique.

    BUG 5  alpha.reshape(-1) needed because Predictive injects a leading
           sample dimension making alpha shape (..., K); W_scale expand
           then fails with a dimension count mismatch.

    BUG 6  W.t() fails when Predictive makes W shape (..., P, K).
           Fix: W.transpose(-2,-1) and torch.matmul instead of torch.mm.

    BUG 7  Mini-batching local latent f: passing batch_size < N means the
           guide is built for N rows but only ever sees batch_size rows →
           shape mismatch at Predictive time.  Fix: pass full X to
           svi.step() every iteration (no mini-batching).
    """
    n_obs    = X.shape[0]
    n_assets = X.shape[1]

    # ── ARD precisions: (K,) ─────────────────────────────────────────────────
    alpha = pyro.sample(
        "alpha",
        dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1)
    )
    # BUG 5 FIX: flatten away any leading sample dims added by Predictive
    alpha_k = alpha.reshape(-1)                         # always (K,)

    # ── Factor loadings W: (P, K) ─────────────────────────────────────────────
    # BUG 1 FIX: single sample site, no nested plates, .to_event(2)
    W_scale = (1.0 / torch.sqrt(alpha_k)).unsqueeze(0).expand(n_assets, n_factors)
    W = pyro.sample(
        "W",
        dist.Normal(torch.zeros(n_assets, n_factors), W_scale).to_event(2)
    )                                                   # declared (P, K)

    # ── Observation noise ─────────────────────────────────────────────────────
    tau   = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)

    # ── Factor scores + likelihood — one plate, unique name ───────────────────
    # BUG 2+3+4 FIX: single plate "time_steps" (≠ any sample site name)
    # covering both f and obs.  No mini-batching (BUG 7 FIX).
    with pyro.plate("time_steps", n_obs):
        f = pyro.sample(
            "f",
            dist.Normal(torch.zeros(n_factors), torch.ones(n_factors)).to_event(1)
        )                                               # (N, K)

        # BUG 6 FIX: transpose(-2,-1) works for (P,K) and (...,P,K)
        #            matmul broadcasts over any leading dims
        mean = torch.matmul(f, W.transpose(-2, -1))    # (N, P)

        pyro.sample(
            "obs",
            dist.Normal(mean, sigma).to_event(1),
            obs=X
        )


def train_svi_model(X, n_factors=5, lr=0.01, iterations=500, batch_size=None):
    """
    Train the Bayesian factor model via SVI and return per-asset scores.

    Args:
        X:          np.ndarray (N, P) — standardised returns matrix
        n_factors:  K — number of latent factors
        lr:         Adam learning rate
        iterations: number of SVI gradient steps
        batch_size: ignored (kept for API compatibility — see BUG 7 above)

    Returns:
        scores: np.ndarray (P,) — sum |posterior mean W| per asset.
                Higher = stronger systematic factor exposure.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)

    pyro.clear_param_store()

    guide     = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": lr})
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # BUG 7 FIX: pass full X every step — no mini-batching of local latent f
    for step in range(iterations):
        loss = svi.step(X_tensor, n_factors)
        if step % 100 == 0:
            print(f"    Step {step:4d}: ELBO loss = {loss:.4f}")

    # ── Posterior W via Predictive ────────────────────────────────────────────
    predictive = Predictive(model, guide=guide, num_samples=200,
                            return_sites=["W"])
    with torch.no_grad():
        posterior = predictive(X_tensor, n_factors)

    # W_samples: (200, P, K)  →  W_mean: (P, K)  →  scores: (P,)
    W_samples = posterior["W"]                          # (num_samples, P, K)
    W_mean    = W_samples.mean(dim=0).numpy()           # (P, K)
    scores    = np.sum(np.abs(W_mean), axis=1)          # (P,)
    return scores
