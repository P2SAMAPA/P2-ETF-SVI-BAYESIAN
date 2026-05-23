import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import numpy as np

# Enable validation
pyro.enable_validation(True)

def model(X, n_factors):
    n_assets = X.shape[1]
    # Priors for factor loadings (ARD: separate precisions per factor)
    alpha = pyro.sample("alpha", dist.Gamma(1.0, 1.0).expand([n_factors]).to_event(1))
    # Factor loadings: shape (n_assets, n_factors) – one batch dimension (factors)
    # We'll use a plate for assets, and inside it, a plate for factors (or vice versa)
    with pyro.plate("assets", n_assets):
        with pyro.plate("factors", n_factors):
            W = pyro.sample("W", dist.Normal(0.0, 1.0 / torch.sqrt(alpha)).to_event(1))
    # Prior for factor scores (standard normal) per observation
    n_obs = X.shape[0]
    with pyro.plate("observations", n_obs):
        with pyro.plate("factors", n_factors):
            f = pyro.sample("f", dist.Normal(0.0, 1.0).to_event(1))
    # Observation noise precision
    tau = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)
    # Likelihood: X = f @ W.T + noise
    # Need to align dimensions: f is (n_obs, n_factors), W is (n_assets, n_factors)
    mean = torch.mm(f, W.t())
    with pyro.plate("data", n_obs):
        pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=X)

# Use auto guide for simplicity (no need to write manual guide)
from pyro.infer.autoguide import AutoDiagonalNormal

def train_svi_model(X, n_factors=5, lr=0.01, iterations=500, batch_size=32):
    n_obs = X.shape[0]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # Set up guide
    guide = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    # Training loop
    for step in range(iterations):
        idx = np.random.choice(n_obs, min(batch_size, n_obs), replace=False)
        batch = X_tensor[idx]
        loss = svi.step(batch, n_factors)
        if step % 100 == 0:
            print(f"    Step {step}: loss = {loss:.4f}")
    # Posterior predictive to get factor loading means
    predictive = Predictive(model, guide=guide, num_samples=500)
    posterior = predictive(X_tensor, n_factors)
    W_mean = posterior["W"].mean(dim=0).numpy()  # (n_assets, n_factors)
    scores = np.sum(np.abs(W_mean), axis=1)
    return scores
