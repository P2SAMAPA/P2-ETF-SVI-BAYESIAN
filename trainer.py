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
    all_results = {}          # → tab 1 JSON (best window per ETF)
    all_windows = {}          # → tab 2 JSON (every window, all ETFs)
    today       = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Bayesian SVI) ===")

        returns = data_manager.prepare_returns_matrix(df, tickers)

        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            all_windows[universe_name] = {"windows": {}}
            continue

        available_tickers = [t for t in tickers if t in returns.columns]
        if not available_tickers:
            print("  No tickers available in returns matrix")
            all_results[universe_name] = {"top_etfs": []}
            all_windows[universe_name] = {"windows": {}}
            continue

        best_per_etf   = {}
        window_results = {}   # used by both tab1 and tab2

        for win in config.WINDOWS:
            if len(returns) < win + 2:
                print(f"  Skipping window {win}d (insufficient data)")
                continue

            print(f"  Processing window {win}d...")

            ret_win = returns[available_tickers].iloc[-win:].values.astype(np.float32)
            col_std = ret_win.std(axis=0)
            col_std[col_std < 1e-8] = 1.0
            ret_win = (ret_win - ret_win.mean(axis=0)) / col_std

            pyro.clear_param_store()

            try:
                scores = train_svi_model(
                    ret_win,
                    n_factors=config.N_FACTORS,
                    lr=config.LEARNING_RATE,
                    iterations=config.N_ITERATIONS,
                    batch_size=config.BATCH_SIZE,
                )
            except Exception as e:
                print(f"  Window {win}d failed: {e}")
                import traceback; traceback.print_exc()
                continue

            if len(scores) != len(available_tickers):
                print(f"  Score/ticker mismatch ({len(scores)} vs {len(available_tickers)}), skipping")
                continue

            score_dict = {available_tickers[i]: float(scores[i])
                          for i in range(len(available_tickers))}
            window_results[win] = score_dict

            for etf, score in score_dict.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        # ── Fallback ──────────────────────────────────────────────────────────
        if not best_per_etf:
            print("  No valid predictions — falling back to historical mean return")
            for etf in available_tickers:
                mean_ret = returns[etf].iloc[-252:].mean()
                if not np.isnan(mean_ret):
                    best_per_etf[etf] = (float(mean_ret), 0)

        if not best_per_etf:
            all_results[universe_name] = {"top_etfs": []}
            all_windows[universe_name] = {"windows": {}}
            continue

        # ── Tab 1: best window per ETF ────────────────────────────────────────
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

        # ── Tab 2: all windows, all ETFs, top-3 per window ───────────────────
        # window_results = {win: {ticker: score, ...}, ...}
        # Build per-window top-N and full ranking for the app to display
        windows_for_tab2 = {}
        for win, score_dict in window_results.items():
            sorted_win = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            windows_for_tab2[str(win)] = {
                "top_etfs": [
                    {"ticker": t, "svi_score": float(s)}
                    for t, s in sorted_win[:config.TOP_N]
                ],
                "full_ranking": [
                    {"ticker": t, "svi_score": float(s)}
                    for t, s in sorted_win
                ],
            }

        all_windows[universe_name] = {
            "windows":   windows_for_tab2,
            "run_date":  today,
        }

    # ── Save tab 1 JSON (existing format, unchanged) ──────────────────────────
    Path("results").mkdir(exist_ok=True)

    tab1_path = Path(f"results/svi_bayesian_{today}.json")
    with open(tab1_path, "w") as f:
        json.dump(
            convert_to_serializable({"run_date": today, "universes": all_results}),
            f, indent=2
        )

    # ── Save tab 2 JSON (new file, window-level detail) ───────────────────────
    tab2_path = Path(f"results/svi_bayesian_windows_{today}.json")
    with open(tab2_path, "w") as f:
        json.dump(
            convert_to_serializable({"run_date": today, "universes": all_windows}),
            f, indent=2
        )

    # ── Push both files ───────────────────────────────────────────────────────
    import push_results
    push_results.push_daily_result(tab1_path)
    push_results.push_daily_result(tab2_path)

    print("\n=== Bayesian SVI Engine complete ===")
    print(f"  Tab 1 file: {tab1_path.name}")
    print(f"  Tab 2 file: {tab2_path.name}")


if __name__ == "__main__":
    main()
