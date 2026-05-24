import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pyro

import config
import data_manager
from svi_bayesian import train_svi_model


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df          = data_manager.load_master_data()
    all_results = {}
    today       = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Bayesian SVI) ===")

        returns = data_manager.prepare_returns_matrix(df, tickers)

        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # FIX: only use tickers that actually exist in returns columns.
        # Original code used `tickers` directly as index into scores, but
        # prepare_returns_matrix may drop tickers with no data, so
        # len(scores) < len(tickers) → "Score/ticker mismatch".
        available_tickers = [t for t in tickers if t in returns.columns]
        if not available_tickers:
            print("  No tickers available in returns matrix")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf   = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue

            print(f"  Processing window {win}d...")

            # Slice only the available tickers — this is what scores will align to
            ret_win = returns[available_tickers].iloc[-win:].values.astype(np.float32)

            # Standardise per asset column
            col_std = ret_win.std(axis=0)
            col_std[col_std < 1e-8] = 1.0
            ret_win = (ret_win - ret_win.mean(axis=0)) / col_std

            # Clear param store between windows — AutoDiagonalNormal caches
            # variational parameter shapes tied to N; stale cache from a
            # different window size causes shape errors on the next window.
            pyro.clear_param_store()

            try:
                scores = train_svi_model(
                    ret_win,
                    n_factors=config.N_FACTORS,
                    lr=config.LEARNING_RATE,
                    iterations=config.N_ITERATIONS,
                    batch_size=config.BATCH_SIZE,   # passed but not used in model
                )
            except Exception as e:
                print(f"  Window {win}d failed: {e}")
                import traceback; traceback.print_exc()
                continue

            # scores is (P,) aligned to available_tickers
            if len(scores) != len(available_tickers):
                print(f"  Score/ticker mismatch ({len(scores)} vs {len(available_tickers)}), skipping")
                continue

            score_dict = {available_tickers[i]: float(scores[i])
                          for i in range(len(available_tickers))}
            window_results[win] = score_dict

            for etf, score in score_dict.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        # Fallback to historical mean if no SVI windows succeeded
        if not best_per_etf:
            print("  No valid predictions — falling back to historical mean return")
            for etf in available_tickers:
                mean_ret = returns[etf].iloc[-252:].mean()
                if not np.isnan(mean_ret):
                    best_per_etf[etf] = (float(mean_ret), 0)

        if not best_per_etf:
            all_results[universe_name] = {"top_etfs": []}
            continue

        full_scores = {
            ticker: {"score": float(score), "best_window": int(win)}
            for ticker, (score, win) in best_per_etf.items()
        }
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs    = [
            {"ticker": ticker, "svi_score": float(score), "best_window": int(win)}
            for ticker, (score, win) in sorted_etfs[:config.TOP_N]
        ]

        print(f"  Top {config.TOP_N} ETFs by Bayesian SVI loading magnitude: "
              f"{[e['ticker'] for e in top_etfs]}")
        for e in top_etfs:
            print(f"    {e['ticker']}: {e['svi_score']:.6f}  (best window: {e['best_window']}d)")

        all_results[universe_name] = {
            "top_etfs":       top_etfs,
            "full_scores":    full_scores,
            "window_results": window_results,
            "run_date":       today,
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/svi_bayesian_{today}.json")
    with open(local_path, "w") as f:
        json.dump(
            convert_to_serializable({"run_date": today, "universes": all_results}),
            f, indent=2
        )

    import push_results
    push_results.push_daily_result(local_path)

    print("\n=== Bayesian SVI Engine complete ===")


if __name__ == "__main__":
    main()
