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


# 4. MAIN

if __name__ == '__main__':
    DATA_DIR = 'data/raw/tennis_atp'

    print("Loading matches...")
    matches = load_matches(DATA_DIR, start_year=2000)

    print("Running Wimbledon backtest (2010-2025)...")
    results = backtest_wimbledon(matches, test_years=range(2010, 2026), blend_w=0.5)

    evaluate(results)
    calibration_plot(results)

    # Save results
    os.makedirs('outputs', exist_ok=True)
    results.to_csv('outputs/backtest_results.csv', index=False)
    print("\nBacktest results saved to outputs/backtest_results.csv")
    