import pandas as pd
import numpy as np
import os
import glob

# 1. LOAD DATA
# Load all yearly CSV files from 2000 onwards and concatenate them into one big match table.
# We skip challenger files as we are only after tour level matches.

def load_matches(data_dir, start_year=2000):
    """Load all yearly CSV files from data_dir from start_year onward and concatenate.

    Files with 'challenger', 'quali', 'Database', or 'ongoing' in the filename are skipped.
    """
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    dfs = []
    for f in files:
        filename = os.path.basename(f)
        # Skip challenger, quali, database, and other non-tour level matches
        if any(x in filename for x in ['challenger', 'quali', 'Database', 'ongoing']):
            continue
        try:
            year = int(filename.replace('.csv', ''))
        except ValueError:
            continue
        if year < start_year:
            continue
        df = pd.read_csv(f, low_memory=False)
        df['year'] = year
        dfs.append(df)

    matches = pd.concat(dfs, ignore_index=True)

    # Sort chronologically - critical for Elo to avoid leakage
    matches = matches.sort_values('tourney_date').reset_index(drop=True)

    # Drop walkovers and retirements with no score
    matches = matches[matches['score'].notna()]
    matches = matches[~matches['score'].str.contains('W/O', na=False)]

    print(f"Loaded {len(matches):,} matches from {start_year} to present")
    return matches


# 2. ELO ENGINE
# We maintain two ratings per player:
# - overall_elo: updated on every match
# - grass_elo:   updated only on grass matches
# Both start at 1500. New players are initialised at 1500 on first appearance.

INITIAL_ELO = 1500

def expected_score(rating_a, rating_b):
    """Probability that player A beats player B given their Elo ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def k_factor(matches_played):
    """
    FiveThirtyEight dynamic K-factor.
    Starts high for new players, stabilises as match count grows.
    Formula: K = 250 / (n + 5)^0.4
    """
    return 250 / ((matches_played + 5) ** 0.4)

def compute_elo(matches):
    """
    Walk through every match in chronological order and update Elo ratings.
    Returns the match dataframe with pre-match Elo columns added,
    plus a dictionary of final ratings for each player.
    """
    # Dictionaries: player_id -> current rating
    overall_elo = {}
    grass_elo = {}

    # Match counters: player_id -> number of matches played (for K-factor)
    overall_count = {}
    grass_count = {}

    # Lists to store pre-match ratings (what the model sees before each match)
    pre_overall_winner = []
    pre_overall_loser = []
    pre_grass_winner = []
    pre_grass_loser = []

    for _, row in matches.iterrows():
        w = row['winner_id']
        l = row['loser_id']
        surface = row.get('surface', None)

        # Initialise players if first appearance
        for pid in [w, l]:
            if pid not in overall_elo:
                overall_elo[pid] = INITIAL_ELO
                overall_count[pid] = 0
            if pid not in grass_elo:
                grass_elo[pid] = INITIAL_ELO
                grass_count[pid] = 0

        # Record PRE-MATCH ratings (before updating)
        pre_overall_winner.append(overall_elo[w])
        pre_overall_loser.append(overall_elo[l])
        pre_grass_winner.append(grass_elo[w])
        pre_grass_loser.append(grass_elo[l])

        # Update OVERALL Elo
        ew = expected_score(overall_elo[w], overall_elo[l])
        kw = k_factor(overall_count[w])
        kl = k_factor(overall_count[l])
        overall_elo[w] += kw * (1 - ew)
        overall_elo[l] += kl * (0 - (1 - ew))
        overall_count[w] += 1
        overall_count[l] += 1

        # Update GRASS Elo (only on grass matches)
        if surface == 'Grass':
            ew_g = expected_score(grass_elo[w], grass_elo[l])
            kw_g = k_factor(grass_count[w])
            kl_g = k_factor(grass_count[l])
            grass_elo[w] += kw_g * (1 - ew_g)
            grass_elo[l] += kl_g * (0 - (1 - ew_g))
            grass_count[w] += 1
            grass_count[l] += 1

    # Attach pre-match ratings to the dataframe
    matches = matches.copy()
    matches['winner_overall_elo'] = pre_overall_winner
    matches['loser_overall_elo'] = pre_overall_loser
    matches['winner_grass_elo'] = pre_grass_winner
    matches['loser_grass_elo'] = pre_grass_loser

    final_ratings = {
        'overall': overall_elo,
        'grass': grass_elo,
        'overall_count': overall_count,
        'grass_count': grass_count,
    }

    return matches, final_ratings


# 3. BLENDED ELO
# Blend overall and grass Elo with weight w.
# w=0 means overall only; w=1 means grass only; w=0.5 is our starting point.

def blend_elo(matches, w=0.5):
    """Add blended Elo columns to the match dataframe."""
    matches = matches.copy()
    matches['winner_blend_elo'] = (
        w * matches['winner_grass_elo'] +
        (1 - w) * matches['winner_overall_elo']
    )
    matches['loser_blend_elo'] = (
        w * matches['loser_grass_elo'] +
        (1 - w) * matches['loser_overall_elo']
    )
    return matches


# 4. WIN PROBABILITY

def win_prob(winner_elo, loser_elo):
    """P(winner beats loser) from blended Elo ratings."""
    return expected_score(winner_elo, loser_elo)


# 5. MAIN

if __name__ == '__main__':
    DATA_DIR = 'data/raw/tennis_atp'

    print("Loading matches...")
    matches = load_matches(DATA_DIR, start_year=2000)

    print("Computing Elo ratings...")
    matches, final_ratings = compute_elo(matches)

    print("Blending Elo (w=0.5)...")
    matches = blend_elo(matches, w=0.5)

    # Quick sanity check: top 10 players by current overall Elo
    player_db = pd.read_csv('data/raw/tennis_atp/ATP_Database.csv', low_memory=False)

    top_overall = sorted(
        final_ratings['overall'].items(), key=lambda x: x[1], reverse=True
    )[:10]

    print("\nTop 10 players by current overall Elo:")
    for pid, elo in top_overall:
        row = player_db[player_db['id'] == pid]
        name = row['player'].values[0] if len(row) else str(pid)
        print(f"  {name}: {elo:.1f}")

    top_grass = sorted(
        final_ratings['grass'].items(), key=lambda x: x[1], reverse=True
    )[:10]

    print("\nTop 10 players by current grass Elo:")
    for pid, elo in top_grass:
        row = player_db[player_db['id'] == pid]
        name = row['player'].values[0] if len(row) else str(pid)
        print(f"  {name}: {elo:.1f}")

    # Save processed matches
    os.makedirs('data/processed', exist_ok=True)
    matches.to_csv('data/processed/matches_with_elo.csv', index=False)
    print("\nSaved processed matches to data/processed/matches_with_elo.csv")
    