import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from elo import load_matches, compute_elo, expected_score

# TOURNAMENT CALIBRATION
# Tests whether our Monte Carlo tournament probabilities are well-calibrated
# against historical Wimbledon outcomes.
#
# For each Wimbledon (2010-2025, excluding 2020):
#   1. Compute pre-tournament Elo (strict cutoff - no leakage)
#   2. Reconstruct the actual R1 draw from match data
#   3. Run 50,000 Monte Carlo simulations
#   4. Compare predicted round-reach probabilities to actual outcomes
#
# This tests something genuinely different from match-level calibration:
# does compounding probabilities over 7 rounds introduce systematic bias?


# 1. RECONSTRUCT DRAW FROM MATCH DATA
# The actual R1 draw for each Wimbledon is implicit in the match data -
# we just need to find all R128/R64 matches and pair them up.

def reconstruct_draw(matches, year, blend_w=0.5, overall_elo=None, grass_elo=None):
    """
    Reconstruct the first-round draw for a given Wimbledon year.
    Returns an ordered list of 128 player IDs in bracket order,
    plus a dict of their blended Elo ratings.
    """
    wim = matches[
        (matches['tourney_name'] == 'Wimbledon') &
        (matches['year'] == year)
    ].copy()

    if len(wim) == 0:
        return None, None

    # Get R1 matches - could be R128 (128 draw) or R64 (64 draw)
    r1 = wim[wim['round'].isin(['R128', 'R64'])].copy()

    if len(r1) == 0:
        return None, None

    # Build bracket as list of [winner, loser] pairs in match order
    # Sort by match_num to ensure consistent bracket ordering
    r1 = r1.sort_values('match_num')
    bracket = []
    for _, row in r1.iterrows():
        bracket.append(row['winner_id'])
        bracket.append(row['loser_id'])

    # Ensure even number of players (complete pairs only)
    if len(bracket) % 2 != 0:
        bracket = bracket[:-1]

    # Pad or trim to nearest power of 2
    n = len(bracket)
    target = 2 ** int(np.log2(n))
    bracket = bracket[:target * 2] if len(bracket) >= target * 2 else bracket

    # Build blended Elo ratings for all players in the draw
    all_players = list(set(bracket))
    ratings = {}
    for pid in all_players:
        w_overall = overall_elo.get(pid, 1500) if overall_elo else 1500
        w_grass = grass_elo.get(pid, 1500) if grass_elo else 1500
        ratings[pid] = blend_w * w_grass + (1 - blend_w) * w_overall

    return bracket, ratings


# 2. MONTE CARLO SIMULATION (same logic as simulate.py but takes a bracket list)

def simulate_draw(bracket, ratings, n_sims=50000, seed=42):
    """
    Run N Monte Carlo simulations of a tournament bracket.
    bracket: ordered list of player IDs (must be a power of 2 in length)
    ratings: dict of player_id -> blended Elo

    Returns dict: player_id -> dict of round-reach probabilities
    """
    np.random.seed(seed)

    # Ensure bracket is a power of 2
    n_players = len(bracket)
    target = 2 ** int(np.log2(n_players))
    bracket = list(bracket[:target])
    n_players = len(bracket)
    n_rounds = int(np.log2(n_players))

    # Get unique players preserving bracket order
    seen = set()
    players = []
    for pid in bracket:
        if pid not in seen:
            players.append(pid)
            seen.add(pid)

    pid_to_idx = {pid: i for i, pid in enumerate(players)}
    elos = np.array([ratings.get(pid, 1500) for pid in players])

    # Convert bracket to indices
    bracket_idx = [pid_to_idx[pid] for pid in bracket]

    # Track how far each player gets in each simulation
    max_round = np.zeros((len(players), n_sims), dtype=np.int8)

    for sim in range(n_sims):
        current = bracket_idx.copy()

        for rnd in range(n_rounds):
            next_round = []
            for i in range(0, len(current), 2):
                p1 = current[i]
                p2 = current[i + 1]
                prob_p1 = 1 / (1 + 10 ** ((elos[p2] - elos[p1]) / 400))
                if np.random.random() < prob_p1:
                    winner = p1
                    loser = p2
                else:
                    winner = p2
                    loser = p1
                next_round.append(winner)
                max_round[winner, sim] = rnd + 1
            current = next_round

    # Convert to probabilities
    probs = {}
    for pid in players:
        idx = pid_to_idx[pid]
        probs[pid] = {
            'p_r2': (max_round[idx] >= 1).mean(),
            'p_r3': (max_round[idx] >= 2).mean(),
            'p_r4': (max_round[idx] >= 3).mean(),
            'p_qf': (max_round[idx] >= 4).mean(),
            'p_sf': (max_round[idx] >= 5).mean(),
            'p_f':  (max_round[idx] >= 6).mean(),
            'p_w':  (max_round[idx] >= 7).mean(),
        }

    return probs

# 3. GET ACTUAL OUTCOMES
# For each player in the draw, record how far they actually got.

def get_actual_outcomes(matches, year):
    """
    For a given Wimbledon year, return a dict of player_id -> rounds reached.
    """
    wim = matches[
        (matches['tourney_name'] == 'Wimbledon') &
        (matches['year'] == year)
    ].copy()

    if len(wim) == 0:
        return {}

    # Map round names to round numbers
    round_map = {
        'R128': 0, 'R64': 1, 'R32': 2, 'R16': 3,
        'QF': 4, 'SF': 5, 'F': 6, 'RR': 0
    }

    # Every winner reached at least the next round
    outcomes = {}
    for _, row in wim.iterrows():
        w = row['winner_id']
        l = row['loser_id']
        rnd = round_map.get(row['round'], 0)

        # Winner advanced past this round
        outcomes[w] = max(outcomes.get(w, 0), rnd + 1)
        # Loser exited at this round
        if l not in outcomes:
            outcomes[l] = rnd

    return outcomes


# 4. CALIBRATION ANALYSIS
# Pool all players across all years and compare predicted vs actual
# round-reach probabilities.

def run_calibration(matches, test_years=None, blend_w=0.5, n_sims=50000):
    """
    Run full tournament calibration across historical Wimbledons.
    Returns a dataframe with one row per (player, year) with predicted
    probabilities and actual outcomes.
    """
    if test_years is None:
        test_years = [y for y in range(2010, 2026) if y != 2020]

    all_rows = []

    for year in test_years:
        print(f"Processing {year}...", end=' ')

        # Get Wimbledon start date for this year
        wim = matches[
            (matches['tourney_name'] == 'Wimbledon') &
            (matches['year'] == year)
        ]
        if len(wim) == 0:
            print("no data")
            continue

        cutoff = pd.to_datetime(wim['tourney_date'].min())

        # Compute Elo strictly before this Wimbledon
        train = matches[
            pd.to_datetime(matches['tourney_date']) < cutoff
        ].copy()
        _, ratings = compute_elo(train)

        # Reconstruct the draw
        bracket, player_ratings = reconstruct_draw(
            matches, year, blend_w=blend_w,
            overall_elo=ratings['overall'],
            grass_elo=ratings['grass']
        )

        if bracket is None:
            print("no draw data")
            continue

        # Simulate tournament
        probs = simulate_draw(bracket, player_ratings, n_sims=n_sims)

        # Get actual outcomes
        outcomes = get_actual_outcomes(matches, year)

        # Build rows
        for pid, pred in probs.items():
            actual = outcomes.get(pid, 0)
            all_rows.append({
                'year': year,
                'player_id': pid,
                'pred_r2': pred['p_r2'],
                'pred_r3': pred['p_r3'],
                'pred_r4': pred['p_r4'],
                'pred_qf': pred['p_qf'],
                'pred_sf': pred['p_sf'],
                'pred_f':  pred['p_f'],
                'pred_w':  pred['p_w'],
                'actual_rounds': actual,
                'reached_r2': int(actual >= 1),
                'reached_r3': int(actual >= 2),
                'reached_r4': int(actual >= 3),
                'reached_qf': int(actual >= 4),
                'reached_sf': int(actual >= 5),
                'reached_f':  int(actual >= 6),
                'reached_w':  int(actual >= 7),
            })

        print(f"{len(probs)} players")

    return pd.DataFrame(all_rows)


# 5. CALIBRATION PLOT
# For each round, bin players by their predicted probability and
# compare to actual reach rate. Perfect calibration = diagonal line.

def plot_calibration(df, output_path='outputs/tournament_calibration.png'):
    """
    Plot calibration curves for QF, SF, Final and Winner.
    """
    os.makedirs('outputs', exist_ok=True)

    rounds = [
        ('pred_qf', 'reached_qf', 'Quarter-Final'),
        ('pred_sf', 'reached_sf', 'Semi-Final'),
        ('pred_f',  'reached_f',  'Final'),
        ('pred_w',  'reached_w',  'Winner'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, (pred_col, actual_col, label) in enumerate(rounds):
        ax = axes[i]

        # Bin by predicted probability
        df['bin'] = pd.cut(df[pred_col], bins=10)
        grouped = df.groupby('bin', observed=True).agg(
            mean_pred=(pred_col, 'mean'),
            mean_actual=(actual_col, 'mean'),
            count=(actual_col, 'count')
        ).dropna()

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax.scatter(
            grouped['mean_pred'], grouped['mean_actual'],
            s=grouped['count'] * 3,
            alpha=0.7, color='steelblue', label='Model'
        )

        # Connect dots to show trend
        ax.plot(
            grouped['mean_pred'], grouped['mean_actual'],
            color='steelblue', alpha=0.4
        )

        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Actual reach rate')
        ax.set_title(f'Calibration: {label}')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle('Tournament Calibration: Wimbledon 2010-2025\n(excluding 2020)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nCalibration plot saved to {output_path}")


# 6. SUMMARY STATISTICS

def print_calibration_summary(df):
    """Print key calibration statistics."""
    print("\n--- Tournament Calibration Summary ---")
    print(f"Total player-years: {len(df)}")
    print(f"Wimbledon editions: {df['year'].nunique()}")

    rounds = [
        ('pred_qf', 'reached_qf', 'QF'),
        ('pred_sf', 'reached_sf', 'SF'),
        ('pred_f',  'reached_f',  'Final'),
        ('pred_w',  'reached_w',  'Winner'),
    ]

    print(f"\n{'Round':<10} {'Mean pred':>10} {'Mean actual':>12} {'Brier':>8}")
    print("-" * 45)

    for pred_col, actual_col, label in rounds:
        from sklearn.metrics import brier_score_loss
        mean_pred = df[pred_col].mean()
        mean_actual = df[actual_col].mean()
        bs = brier_score_loss(df[actual_col], df[pred_col])
        print(f"{label:<10} {mean_pred:>10.3f} {mean_actual:>12.3f} {bs:>8.4f}")

    # Check for systematic overconfidence on favourites
    print("\n--- Top-10 predicted winners: actual vs predicted ---")
    top10 = df.nlargest(10, 'pred_w')[['year', 'player_id', 'pred_w', 'reached_w']]
    print(top10.to_string(index=False))


# 7. MAIN

if __name__ == '__main__':
    DATA_DIR = 'data/raw/tennis_atp'

    print("Loading matches...")
    matches = load_matches(DATA_DIR, start_year=2000)
    matches['tourney_date'] = pd.to_datetime(matches['tourney_date'])

    print("\nRunning tournament calibration (this takes ~10 minutes)...")
    df = run_calibration(matches, n_sims=10000)  # 10k sims for speed

    print_calibration_summary(df)
    plot_calibration(df)

    os.makedirs('outputs', exist_ok=True)
    df.to_csv('outputs/tournament_calibration.csv', index=False)
    print("\nCalibration data saved to outputs/tournament_calibration.csv")