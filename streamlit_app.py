import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Bayesian SVI", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
.sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
.universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem;
                  padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
.etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white;
            border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker { font-size: 1.3rem; font-weight: bold; }
.etf-score  { font-size: 0.9rem; margin-top: 0.3rem; }
.win-card   { background: linear-gradient(135deg, #16a085 0%, #1a5276 100%); color: white;
              border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center;
              box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Bayesian SVI (Stochastic Variational Inference) Engine</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Full Bayesian factor model | ARD priors | SVI with AutoNormal guide | '
            'Factor loading magnitude = score | Multi‑window evaluation</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧠 Bayesian SVI")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Factors:** {config.N_FACTORS} | **Iterations:** {config.N_ITERATIONS}")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016, 4032 days")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN    = config.HF_TOKEN


# ── HuggingFace helpers ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                           detail=True, recursive=True)
                 if f["type"] == "file"]
        return files
    except Exception as e:
        return [f"Error: {e}"]


def find_latest_json(files, prefix):
    """Return the most recent JSON whose basename starts with `prefix`."""
    matches = [f for f in files if f.endswith(".json") and prefix in f]
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0]


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ── Load data ─────────────────────────────────────────────────────────────────
files = list_repo_files()

latest_tab1 = find_latest_json(files, "svi_bayesian_2")       # e.g. svi_bayesian_2026-05-24.json
latest_tab2 = find_latest_json(files, "svi_bayesian_windows_") # e.g. svi_bayesian_windows_2026-05-24.json

if not latest_tab1:
    st.error("No results found. Run trainer first.")
    st.stop()

data_tab1 = load_json(latest_tab1)
if "error" in data_tab1:
    st.error(f"Error loading tab 1 data: {data_tab1['error']}")
    st.stop()

st.session_state["run_date"] = data_tab1["run_date"]
universes_tab1 = data_tab1["universes"]

# Tab 2 data is optional — show tab 2 only if the window file exists
data_tab2      = load_json(latest_tab2) if latest_tab2 else None
universes_tab2 = data_tab2["universes"] if data_tab2 and "error" not in data_tab2 else None


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — existing view, best loading across all windows (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs by Bayesian SVI Loading Magnitude")

    with st.expander("📖 Interpretation", expanded=True):
        st.markdown("""
- **Bayesian factor model** with Automatic Relevance Determination (ARD) priors on factor loadings.
- **Stochastic Variational Inference (SVI)** approximates the posterior using a mean‑field guide.
- The **loading magnitude** for each ETF is the sum of absolute posterior means of its factor loadings.
- Higher loading → stronger systematic risk exposure → often higher expected return.
- For each ETF, the rolling window that gives the **highest loading magnitude** is selected.
- This is a full Bayesian treatment, not a PCA approximation.
        """)

    for universe_name, uni_data in universes_tab1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue

        st.markdown(
            f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>',
            unsafe_allow_html=True
        )
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">loading = {etf['svi_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for ticker, info in full.items():
                    if isinstance(info, dict):
                        score = info.get("score", 0.0)
                        win   = info.get("best_window", "N/A")
                    else:
                        score = info
                        win   = "N/A"
                    rows.append({"ETF": ticker, "Loading": score, "Best Window": win})
                df = pd.DataFrame(rows)
                df["Loading"] = pd.to_numeric(df["Loading"], errors="coerce")
                df = df.dropna(subset=["Loading"]).sort_values("Loading", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption("Full Bayesian factor model with SVI (Pyro). "
               "Higher loading → stronger factor exposure → overweight signal.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — window explorer: choose a window, see all three universes ranked
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore Rankings by Window")

    if universes_tab2 is None:
        st.warning(
            "Window-level detail file not found. "
            "Re-run the trainer to generate `svi_bayesian_windows_<date>.json`."
        )
        st.stop()

    with st.expander("📖 How to use this tab", expanded=False):
        st.markdown("""
- Select a **lookback window** from the dropdown to see how ETF rankings change.
- Each universe (FI/Commodities, Equity Sectors, Combined) is shown side by side.
- The **Top 3** cards show the highest-loading ETFs **at that specific window**.
- The full table below shows every ETF's loading score for that window.
- Tip: compare 252d vs 504d to see which picks are stable across time horizons.
        """)

    # Collect all windows available across universes
    all_available_wins = set()
    for uni_data in universes_tab2.values():
        all_available_wins.update(uni_data.get("windows", {}).keys())

    win_options = sorted([int(w) for w in all_available_wins])
    win_labels  = {w: f"{w}d  (~{round(w/252, 1)} yr)" for w in win_options}

    if not win_options:
        st.error("No window data available.")
        st.stop()

    # Default to 252d if available, else first option
    default_idx = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: win_labels[w],
    )
    win_key = str(selected_win)

    st.markdown(f"### Rankings at **{selected_win}d** window across all universes")
    st.markdown("---")

    universe_order = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
    universe_labels = {
        "FI_COMMODITIES": "🏦 FI & Commodities",
        "EQUITY_SECTORS": "📈 Equity Sectors",
        "COMBINED":       "🌐 Combined",
    }

    for universe_name in universe_order:
        uni_data = universes_tab2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        label = universe_labels.get(universe_name,
                                    universe_name.replace("_", " ").title())
        st.markdown(
            f'<div class="universe-title">{label}</div>',
            unsafe_allow_html=True
        )

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d window.")
            st.divider()
            continue

        # Top 3 cards
        top_etfs = win_data.get("top_etfs", [])
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">loading = {etf['svi_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
</div>
""", unsafe_allow_html=True)

        # Full ranking table
        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            full_ranking = win_data.get("full_ranking", [])
            if full_ranking:
                df = pd.DataFrame(full_ranking)
                df.columns = ["ETF", "Loading"]
                df["Loading"] = pd.to_numeric(df["Loading"], errors="coerce")
                df = df.dropna(subset=["Loading"]).sort_values("Loading", ascending=False)
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No ranking data for this window.")

        st.divider()

    st.caption(
        f"Showing SVI factor loading magnitudes for the {selected_win}d window. "
        "Run date: " + (data_tab2.get("run_date", "unknown"))
    )
