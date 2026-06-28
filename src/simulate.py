import pandas as pd
import numpy as np
import os
from elo import load_matches, compute_elo, expected_score

# 1. LOAD THE 2026 DRAW
# The draw is a CSV with 128 players in bracket order.
# Slots 1-2 play each other, slots 3-4 play each other, etc.
# We will create this file manually from the official Wimbledon draw.

def load_draw(draw_path):
    """Load the 2026 Wimbledon draw from CSV."""
    draw = pd.read_csv(draw_path)
    print(f"Loaded draw with {len(draw)} players")
    return draw


# 2. BUILD PLAYER RATINGS
# For each player in the draw, look up their pre-tournament
# overall and grass Elo from the ratings computed on all
# matches up to the draw date (26 June 2026).

def get_player_ratings(draw, final_ratings, blend_w=0.5):
    """
    For each player in the draw, compute their blended Elo rating.
    Players not seen in the training data get the default of 1500.
    """
    overall_elo = final_ratings['overall']
    grass_elo = final_ratings['grass']

    ratings = []
    for _, row in draw.iterrows():
        pid = row['player_id']
        name = row['player_name']

        w_overall = overall_elo.get(pid, 1500)
        w_grass = grass_elo.get(pid, 1500)
        w_blend = blend_w * w_grass + (1 - blend_w) * w_overall

        ratings.append({
            'player_id': pid,
            'player_name': name,
            'overall_elo': w_overall,
            'grass_elo': w_grass,
            'blend_elo': w_blend,
            'slot': row['slot']
        })

    ratings_df = pd.DataFrame(ratings)
    return ratings_df


# 3. MONTE CARLO SIMULATION
# Simulate the 128-player single elimination bracket N times.
# In each simulation, for every match we draw a random number
# and compare it to the favourite's win probability.
# We tally how often each player reaches each round.

def simulate_tournament(ratings_df, n_sims=50000, seed=42):
    """
    Run N Monte Carlo simulations of the Wimbledon draw.
    Returns a dataframe with each player's probability of
    reaching R16, QF, SF, Final, and winning the title.
    """
    np.random.seed(seed)

    n_players = len(ratings_df)
    assert n_players == 128, f"Expected 128 players, got {n_players}"

    # Store Elo ratings as a numpy array for speed
    elos = ratings_df['blend_elo'].values

    # Tally matrix: rows = players, cols = rounds reached
    # Rounds: R128(0), R64(1), R32(2), R16(3), QF(4), SF(5), F(6), W(7)
    round_counts = np.zeros((n_players, 8), dtype=int)

    # All players start in the draw (round 0)
    round_counts[:, 0] = n_sims

    for sim in range(n_sims):
        # Current players in bracket order (indices into ratings_df)
        bracket = list(range(n_players))

        for rnd in range(7):  # 7 rounds to get from 128 to 1
            next_bracket = []
            for i in range(0, len(bracket), 2):
                p1 = bracket[i]
                p2 = bracket[i + 1]

                # Win probability for p1
                elo1 = elos[p1]
                elo2 = elos[p2]
                prob_p1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))

                # Simulate match outcome
                if np.random.random() < prob_p1:
                    winner = p1
                else:
                    winner = p2

                next_bracket.append(winner)
                # Record that winner reached next round
                round_counts[winner, rnd + 1] += 1

            bracket = next_bracket

    # Convert tallies to probabilities
    probs = round_counts / n_sims

    results = ratings_df.copy()
    results['p_r64'] = probs[:, 1]    # Reach R64 (win R128)
    results['p_r32'] = probs[:, 2]    # Reach R32
    results['p_r16'] = probs[:, 3]    # Reach R16
    results['p_qf'] = probs[:, 4]     # Reach QF
    results['p_sf'] = probs[:, 5]     # Reach SF
    results['p_final'] = probs[:, 6]  # Reach Final
    results['p_win'] = probs[:, 7]    # Win title

    return results


# 4. PRINT RESULTS TABLE

def print_results(results, top_n=32):
    """Print the tournament probability table, sorted by title probability."""
    results = results.sort_values('p_win', ascending=False).head(top_n)

    print(f"\n{'Player':<30} {'Elo':>7} {'R16':>6} {'QF':>6} {'SF':>6} {'F':>6} {'Win':>6}")
    print("-" * 70)
    for _, row in results.iterrows():
        print(
            f"{row['player_name']:<30} "
            f"{row['blend_elo']:>7.0f} "
            f"{row['p_r16']:>6.1%} "
            f"{row['p_qf']:>6.1%} "
            f"{row['p_sf']:>6.1%} "
            f"{row['p_final']:>6.1%} "
            f"{row['p_win']:>6.1%}"
        )


# 5. MAIN

if __name__ == '__main__':
    DATA_DIR = 'data/raw/tennis_atp'
    DRAW_PATH = 'data/processed/draw_2026.csv'
    BLEND_W = 0.35
    N_SIMS = 50000

    # Train Elo on all matches up to 26 June 2026
    print("Loading matches...")
    matches = load_matches(DATA_DIR, start_year=2000)

    print("Computing Elo ratings...")
    _, final_ratings = compute_elo(matches)

    # Load the draw and compute player ratings
    print("Loading 2026 Wimbledon draw...")
    draw = load_draw(DRAW_PATH)

    print("Computing player ratings...")
    ratings_df = get_player_ratings(draw, final_ratings, blend_w=BLEND_W)

    # Run simulation
    print(f"Running {N_SIMS:,} Monte Carlo simulations...")
    results = simulate_tournament(ratings_df, n_sims=N_SIMS)

    # Print and save
    print_results(results)

    os.makedirs('outputs', exist_ok=True)
    results.sort_values('p_win', ascending=False).to_csv(
        'outputs/wimbledon_2026_forecast.csv', index=False
    )
    print("\nForecast saved to outputs/wimbledon_2026_forecast.csv")