import pandas as pd
import numpy as np
import os
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
from elo import load_matches, compute_elo, blend_elo, expected_score

# 1. BACKTEST
# For each Wimbledon from 2010 to 2025, we:
# - Compute Elo using only matches BEFORE that Wimbledon (no leakage)
# - Predict every main draw match using blended Elo
# - Record the predicted probability and actual outcome
# - Evaluate with log loss and Brier score

def get_wimbledon_matches(matches, year):
    """Extract all Wimbledon main draw matches for a given year."""
    wim = matches[
        (matches['tourney_name'] == 'Wimbledon') &
        (matches['year'] == year)
    ].copy()
    return wim


def backtest_wimbledon(matches, test_years=range(2010, 2026), blend_w=0.5):
    """
    Walk-forward backtest over historical Wimbledons.
    For each test year, train Elo on all matches before Wimbledon starts,
    then predict every Wimbledon match that year.
    """
    all_results = []

    for year in test_years:
        # Get the date Wimbledon started that year
        wim_matches = get_wimbledon_matches(matches, year)
        if len(wim_matches) == 0:
            print(f"{year}: no Wimbledon matches found, skipping")
            continue

        wim_start = wim_matches['tourney_date'].min()

        # Train Elo on all matches strictly before Wimbledon
        train = matches[matches['tourney_date'] < wim_start].copy()

        if len(train) == 0:
            print(f"{year}: no training data, skipping")
            continue

        # Compute Elo on training data only
        _, ratings = compute_elo(train)

        overall_elo = ratings['overall']
        grass_elo = ratings['grass']

        # Predict each Wimbledon match
        for _, row in wim_matches.iterrows():
            w = row['winner_id']
            l = row['loser_id']

            # Get pre-tournament ratings (1500 if never seen before)
            w_overall = overall_elo.get(w, 1500)
            l_overall = overall_elo.get(l, 1500)
            w_grass = grass_elo.get(w, 1500)
            l_grass = grass_elo.get(l, 1500)

            # Blend ratings
            w_blend = blend_w * w_grass + (1 - blend_w) * w_overall
            l_blend = blend_w * l_grass + (1 - blend_w) * l_overall

            # Predicted probability that winner beats loser
            prob_winner = expected_score(w_blend, l_blend)

            all_results.append({
                'year': year,
                'round': row['round'],
                'winner': row['winner_name'],
                'loser': row['loser_name'],
                'prob_winner': prob_winner,
                'correct': 1 if prob_winner > 0.5 else 0
            })

    results = pd.DataFrame(all_results)
    return results


# 2. METRICS
# Evaluate the backtest results with log loss, Brier score and accuracy.
# We also compare against a naive 50/50 baseline.

def evaluate(results):
    """Print evaluation metrics for backtest results."""

    # True outcome is always 1 (winner won) - we just vary the predicted prob
    y_true = np.ones(len(results))
    y_pred = results['prob_winner'].values

    # Clip probabilities away from 0 and 1 to avoid infinite log loss
    y_pred_clipped = np.clip(y_pred, 0.001, 0.999)

    ll = log_loss(y_true, y_pred_clipped, labels=[0, 1])
    bs = brier_score_loss(y_true, y_pred_clipped)
    acc = results['correct'].mean()

    # Naive baseline: always predict 0.5
    ll_baseline = log_loss(y_true, np.full(len(y_true), 0.5), labels=[0, 1])
    bs_baseline = brier_score_loss(y_true, np.full(len(y_true), 0.5))

    print(f"\n--- Backtest Results (Wimbledon 2010-2025) ---")
    print(f"Matches evaluated: {len(results)}")
    print(f"Accuracy:    {acc:.3f}  (baseline: 0.500)")
    print(f"Log loss:    {ll:.4f}  (baseline: {ll_baseline:.4f})")
    print(f"Brier score: {bs:.4f}  (baseline: {bs_baseline:.4f})")

    print(f"\n--- Results by year ---")
    for year, grp in results.groupby('year'):
        y_t = np.ones(len(grp))
        y_p = np.clip(grp['prob_winner'].values, 0.001, 0.999)
        ll_y = log_loss(y_t, y_p, labels=[0, 1])
        acc_y = grp['correct'].mean()
        print(f"  {year}: {len(grp)} matches | acc={acc_y:.3f} | log loss={ll_y:.4f}")

    return ll, bs, acc


# 3. CALIBRATION PLOT
# Shows whether predicted probabilities match actual win rates.
# A perfectly calibrated model sits on the diagonal.

def calibration_plot(results, output_path='outputs/calibration.png'):
    """Plot calibration curve and save to outputs folder."""
    os.makedirs('outputs', exist_ok=True)

    # Bin predictions into 10 buckets
    results = results.copy()
    results['bin'] = pd.cut(results['prob_winner'], bins=10)
    grouped = results.groupby('bin', observed=True)['correct'].agg(['mean', 'count'])
    bin_centers = [interval.mid for interval in grouped.index]

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(bin_centers, grouped['mean'], s=grouped['count'],
                alpha=0.7, color='steelblue', label='Model')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual win rate')
    plt.title('Calibration plot: Wimbledon backtest 2010-2025')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nCalibration plot saved to {output_path}")


# 4. BLEND WEIGHT SENSITIVITY ANALYSIS
# Grid search over blend weight w from 0 to 1.
# w=0 means overall Elo only, w=1 means grass Elo only.
# We minimise log loss on the full backtest to find the optimal blend.

def blend_weight_search(matches, test_years=range(2010, 2026), weights=None):
    """
    Grid search over blend weights and evaluate each on the Wimbledon backtest.
    Returns a dataframe of results sorted by log loss.
    """
    if weights is None:
        weights = np.arange(0.0, 1.05, 0.05)

    rows = []

    for w in weights:
        all_results = []

        for year in test_years:
            wim_matches = get_wimbledon_matches(matches, year)
            if len(wim_matches) == 0:
                continue

            wim_start = wim_matches['tourney_date'].min()
            train = matches[matches['tourney_date'] < wim_start].copy()
            _, ratings = compute_elo(train)

            overall_elo = ratings['overall']
            grass_elo = ratings['grass']

            for _, row in wim_matches.iterrows():
                wid = row['winner_id']
                lid = row['loser_id']

                w_blend = w * grass_elo.get(wid, 1500) + (1 - w) * overall_elo.get(wid, 1500)
                l_blend = w * grass_elo.get(lid, 1500) + (1 - w) * overall_elo.get(lid, 1500)

                prob_winner = expected_score(w_blend, l_blend)
                correct = 1 if prob_winner > 0.5 else 0

                all_results.append({
                    'prob_winner': prob_winner,
                    'correct': correct
                })

        results_df = pd.DataFrame(all_results)
        y_true = np.ones(len(results_df))
        y_pred = np.clip(results_df['prob_winner'].values, 0.001, 0.999)

        ll = log_loss(y_true, y_pred, labels=[0, 1])
        bs = brier_score_loss(y_true, y_pred)
        acc = results_df['correct'].mean()

        rows.append({'blend_w': round(w, 2), 'log_loss': ll, 'brier': bs, 'accuracy': acc})
        print(f"  w={w:.2f}: log_loss={ll:.4f}, acc={acc:.3f}")

    return pd.DataFrame(rows)


def plot_blend_search(results_df, output_path='outputs/blend_weight_search.png'):
    """Plot log loss and accuracy against blend weight."""
    os.makedirs('outputs', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(results_df['blend_w'], results_df['log_loss'],
             color='steelblue', marker='o', markersize=4)
    best_idx = results_df['log_loss'].idxmin()
    best_w = results_df.loc[best_idx, 'blend_w']
    ax1.axvline(x=best_w, color='red', linestyle='--', alpha=0.7,
                label=f'Optimal w={best_w:.2f}')
    ax1.set_xlabel('Blend weight (w)\n0 = overall Elo only, 1 = grass Elo only')
    ax1.set_ylabel('Log loss')
    ax1.set_title('Log loss vs blend weight')
    ax1.legend()

    ax2.plot(results_df['blend_w'], results_df['accuracy'],
             color='steelblue', marker='o', markersize=4)
    ax2.axvline(x=best_w, color='red', linestyle='--', alpha=0.7,
                label=f'Optimal w={best_w:.2f}')
    ax2.set_xlabel('Blend weight (w)\n0 = overall Elo only, 1 = grass Elo only')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs blend weight')
    ax2.legend()

    plt.suptitle('Blend Weight Sensitivity: Wimbledon 2010-2025', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Blend weight plot saved to {output_path}")


# 5. MAIN

if __name__ == '__main__':
    DATA_DIR = 'data/raw/tennis_atp'

    print("Loading matches...")
    matches = load_matches(DATA_DIR, start_year=2000)

    print("Running Wimbledon backtest (2010-2025)...")
    results = backtest_wimbledon(matches, test_years=range(2010, 2026), blend_w=0.5)
    evaluate(results)
    calibration_plot(results)

    os.makedirs('outputs', exist_ok=True)
    results.to_csv('outputs/backtest_results.csv', index=False)

    print("\nRunning blend weight grid search...")
    print("(Testing 21 values of w from 0.0 to 1.0)")
    blend_results = blend_weight_search(matches, test_years=range(2010, 2026))

    print("\n--- Blend Weight Results ---")
    print(blend_results.to_string(index=False))

    best = blend_results.loc[blend_results['log_loss'].idxmin()]
    print(f"\nOptimal blend weight: w={best['blend_w']:.2f}")
    print(f"Best log loss: {best['log_loss']:.4f}")
    print(f"Best accuracy: {best['accuracy']:.3f}")

    plot_blend_search(blend_results)
    blend_results.to_csv('outputs/blend_weight_search.csv', index=False)
    print("Blend weight results saved to outputs/blend_weight_search.csv")