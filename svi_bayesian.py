import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal
import numpy as np

pyro.enable_validation(True)


def model(X, n_factors):
    n_obs    = X.shape[0]
    n_assets = X.shape[1]

    alpha = pyro.sample(
        "alpha",
        dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1)
    )
    alpha_k = alpha.reshape(-1)

    W_scale = (1.0 / torch.sqrt(alpha_k)).unsqueeze(0).expand(n_assets, n_factors)
    W = pyro.sample(
        "W",
        dist.Normal(torch.zeros(n_assets, n_factors), W_scale).to_event(2)
    )

    tau   = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)

    with pyro.plate("time_steps", n_obs):
        f = pyro.sample(
            "f",
            dist.Normal(torch.zeros(n_factors), torch.ones(n_factors)).to_event(1)
        )
        mean = torch.matmul(f, W.transpose(-2, -1))
        pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=X)


def train_svi_model(X, n_factors=5, lr=0.01, iterations=500, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    pyro.clear_param_store()
    guide     = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": lr})
    svi       = SVI(model, guide, optimizer, loss=Trace_ELBO())
    for step in range(iterations):
        loss = svi.step(X_tensor, n_factors)
        if step % 100 == 0:
            print(f"    Step {step:4d}: ELBO loss = {loss:.4f}")
    predictive = Predictive(model, guide=guide, num_samples=200, return_sites=["W"])
    with torch.no_grad():
        posterior = predictive(X_tensor, n_factors)
    W_mean = posterior["W"].mean(dim=0).squeeze().numpy()  # remove any size-1 leading dims → (P, K)
    scores  = np.abs(W_mean).sum(axis=-1).flatten()         # sum over K dim → (P,) always
   
    return scores
