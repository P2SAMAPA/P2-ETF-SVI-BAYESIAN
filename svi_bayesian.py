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
    with pyro.plate("factors", n_factors):
        with pyro.plate("assets", n_assets):
            W = pyro.sample("W", dist.Normal(0.0, 1.0 / torch.sqrt(alpha)).to_event(2))
    # Prior for factor scores (standard normal)
    n_obs = X.shape[0]
    with pyro.plate("observations", n_obs):
        f = pyro.sample("f", dist.Normal(0.0, 1.0).expand([n_factors]).to_event(1))
    # Observation noise
    tau = pyro.sample("tau", dist.Gamma(1.0, 1.0))
    sigma = 1.0 / torch.sqrt(tau)
    # Likelihood
    mean = torch.mm(f, W.t())
    with pyro.plate("data", n_obs):
        pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=X)

def guide(X, n_factors):
    n_assets = X.shape[1]
    # Guide for alpha
    alpha_loc = pyro.param("alpha_loc", torch.ones(n_factors))
    alpha_scale = pyro.param("alpha_scale", torch.ones(n_factors), constraint=dist.constraints.positive)
    pyro.sample("alpha", dist.Gamma(alpha_loc, alpha_scale).to_event(1))
    # Guide for W
    W_loc = pyro.param("W_loc", torch.randn(n_assets, n_factors))
    W_scale = pyro.param("W_scale", torch.ones(n_assets, n_factors), constraint=dist.constraints.positive)
    with pyro.plate("factors", n_factors):
        with pyro.plate("assets", n_assets):
            pyro.sample("W", dist.Normal(W_loc, W_scale).to_event(2))
    # Guide for f
    n_obs = X.shape[0]
    f_loc = pyro.param("f_loc", torch.randn(n_obs, n_factors))
    f_scale = pyro.param("f_scale", torch.ones(n_obs, n_factors), constraint=dist.constraints.positive)
    with pyro.plate("observations", n_obs):
        pyro.sample("f", dist.Normal(f_loc, f_scale).to_event(1))
    # Guide for tau
    tau_loc = pyro.param("tau_loc", torch.tensor(1.0))
    tau_scale = pyro.param("tau_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    pyro.sample("tau", dist.Gamma(tau_loc, tau_scale))

def train_svi_model(X, n_factors=5, lr=0.01, iterations=500, batch_size=32):
    n_obs = X.shape[0]
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # Set up SVI
    optimizer = Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    # Training loop
    for step in range(iterations):
        # Mini‑batch
        idx = np.random.choice(n_obs, min(batch_size, n_obs), replace=False)
        batch = X_tensor[idx]
        loss = svi.step(batch, n_factors)
        if step % 100 == 0:
            print(f"    Step {step}: loss = {loss:.4f}")
    # Posterior predictive to get factor loading means
    predictive = Predictive(model, guide=guide, num_samples=500)
    posterior = predictive(X_tensor, n_factors)
    # Posterior mean of loadings
    W_mean = posterior["W"].mean(dim=0).numpy()  # (n_assets, n_factors)
    # Score: sum of absolute loadings across factors
    scores = np.sum(np.abs(W_mean), axis=1)
    return scores

def predict_svi(X, n_factors=5):
    # Placeholder: return the scores (already computed in training)
    # For a pure prediction, we would need to sample f and compute mean.
    return None
