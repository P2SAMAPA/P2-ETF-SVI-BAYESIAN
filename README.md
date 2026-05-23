# Bayesian SVI Engine

Full Bayesian factor model with Automatic Relevance Determination (ARD) priors. Inference via Stochastic Variational Inference (SVI) using Pyro. The loading magnitude (sum of absolute posterior means of factor loadings) is the score – higher loading indicates stronger systematic exposure. Multi‑window evaluation selects the best window per ETF.

- **Model:** Probabilistic factor model with Normal prior on loadings (ARD), Gamma prior on precision
- **Inference:** SVI with AutoNormal guide (Pyro)
- **Factors:** 5 (configurable)
- **Windows:** 63, 252, 504, 1008, 2016, 4032 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
