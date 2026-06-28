import pandas as pd
import numpy as np
import os
from elo import load_matches

# 1. SERVE STATISTICS
# For each player we compute their grass-court serve quality from
# historical matches. Grass amplifies serve dominance more than any
# other surface, so this is the single most important feature to add
# on top of Elo for a Wimbledon-specific model.
#
# CRITICAL: every statistic for a match on date D must use ONLY
# matches played strictly before D. This prevents temporal leakage,
# which is the cardinal sin of any backtested forecasting model.

def build_serve_history(matches):
    """
    Reshape the match data into a per-player, per-match long format
    so that each row is one player's serve performance in one match.
    This makes it easy to compute rolling pre-match averages later.

    Returns a dataframe with one row per (player, match) with their
    serve stats and the match date.
    """
    # Winner rows
    winners = matches[[
        'tourney_date', 'surface', 'winner_id',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon'
    ]].copy()
    winners.columns = [
        'date', 'surface', 'player_id',
        'ace', 'df', 'svpt', 'firstIn', 'firstWon', 'secondWon'
    ]

    # Loser rows
    losers = matches[[
        'tourney_date', 'surface', 'loser_id',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon'
    ]].copy()
    losers.columns = [
        'date', 'surface', 'player_id',
        'ace', 'df', 'svpt', 'firstIn', 'firstWon', 'secondWon'
    ]

    # Stack winner and loser rows into one long table
    serve = pd.concat([winners, losers], ignore_index=True)

    # Drop rows with missing serve data (older matches, walkovers)
    serve = serve.dropna(subset=['svpt'])
    serve = serve[serve['svpt'] > 0]

  # Ensure date is datetime type
    serve['date'] = pd.to_datetime(serve['date'])

    # Sort chronologically
    serve = serve.sort_values('date').reset_index(drop=True)

    return serve


def grass_serve_rating(serve_history, player_id, cutoff_date, min_matches=5):
    """
    Compute a player's grass serve quality using ONLY grass matches
    played strictly before cutoff_date.

    Returns a single serve score combining first-serve points won and
    ace rate, or None if the player has too few grass matches.
    """
    # Filter to this player's grass matches before the cutoff
    hist = serve_history[
        (serve_history['player_id'] == player_id) &
        (serve_history['surface'] == 'Grass') &
        (serve_history['date'] < cutoff_date)
    ]

    if len(hist) < min_matches:
        return None

    # Aggregate totals (not averages of ratios - we sum then divide
    # so that high-volume matches are weighted correctly)
    total_svpt = hist['svpt'].sum()
    total_ace = hist['ace'].sum()
    total_firstIn = hist['firstIn'].sum()
    total_firstWon = hist['firstWon'].sum()
    total_secondWon = hist['secondWon'].sum()

    if total_svpt == 0 or total_firstIn == 0:
        return None

    # Serve performance metrics
    ace_rate = total_ace / total_svpt
    first_serve_pct = total_firstIn / total_svpt
    first_won_pct = total_firstWon / total_firstIn
    second_won_pct = total_secondWon / (total_svpt - total_firstIn) if (total_svpt - total_firstIn) > 0 else 0

    # Combined serve score: overall points won on serve
    serve_points_won = (total_firstWon + total_secondWon) / total_svpt

    return {
        'serve_points_won': serve_points_won,
        'ace_rate': ace_rate,
        'first_serve_pct': first_serve_pct,
        'first_won_pct': first_won_pct,
        'second_won_pct': second_won_pct,
        'grass_matches': len(hist)
    }


# 2. RECENT GRASS FORM
# Win rate on grass over the last 12 months before the cutoff date.
# Captures players who are in form on grass right now (e.g. Queen's
# and Halle 2026 results), which Elo only reflects with a lag.

def recent_grass_form(matches, player_id, cutoff_date, months=12):
    """
    Win rate on grass in the months before cutoff_date.
    Returns win rate and match count, or None if no recent grass matches.
    """
    matches = matches.copy()
    matches['tourney_date'] = pd.to_datetime(matches['tourney_date'])
    window_start = cutoff_date - pd.DateOffset(months=months)

    # Grass matches in the window before cutoff
    window = matches[
        (matches['surface'] == 'Grass') &
        (matches['tourney_date'] < cutoff_date) &
        (matches['tourney_date'] >= window_start)
    ]

    wins = (window['winner_id'] == player_id).sum()
    losses = (window['loser_id'] == player_id).sum()
    total = wins + losses

    if total == 0:
        return None

    return {
        'recent_grass_winrate': wins / total,
        'recent_grass_matches': total
    }


# 3. MAIN (test the functions)

if __name__ == '__main__':
    DATA_DIR = 'data/raw/tennis_atp'

    print("Loading matches...")
    matches = load_matches(DATA_DIR, start_year=2000)

    print("Building serve history...")
    serve_history = build_serve_history(matches)
    print(f"Serve history: {len(serve_history):,} player-match rows")

    # Test on a known big server before Wimbledon 2025
    cutoff = pd.Timestamp('2025-06-30')

    # Find John Isner's ID as a test (famous big server)
    db = pd.read_csv('data/raw/tennis_atp/ATP_Database.csv', low_memory=False)
    test_players = ['Sinner', 'Djokovic', 'Fritz', 'Shelton']

    print(f"\nGrass serve ratings (using only data before {cutoff.date()}):")
    for surname in test_players:
        hit = db[db['player'].str.contains(surname, na=False, case=False)]
        if len(hit) == 0:
            continue
        pid = hit['id'].values[0]
        name = hit['player'].values[0]
        serve = grass_serve_rating(serve_history, pid, cutoff)
        form = recent_grass_form(matches, pid, cutoff)
        if serve:
            print(f"  {name}: serve_pts_won={serve['serve_points_won']:.3f}, "
                  f"ace_rate={serve['ace_rate']:.3f}, grass_matches={serve['grass_matches']}")
        if form:
            print(f"    recent grass winrate={form['recent_grass_winrate']:.3f} "
                  f"({form['recent_grass_matches']} matches)")
            