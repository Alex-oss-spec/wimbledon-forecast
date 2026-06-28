import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from elo import load_matches, compute_elo, expected_score
from features import build_serve_history, grass_serve_rating, recent_grass_form

# 1. BUILD THE FEATURE TABLE
# For each Wimbledon match in our test years, we compute a feature
# vector based ONLY on data available before that year's Wimbledon.
#
# Features are all DIFFERENCES between the two players:
#   - elo_diff:        blended Elo difference
#   - serve_diff:      grass serve-points-won difference
#   - form_diff:       recent grass win-rate difference
#
# To avoid the model trivially learning "player A always wins", we
# randomly flip half the matches so the label is balanced 50/50.

def build_feature_table(matches, test_years=range(2010, 2026), blend_w=0.5, seed=42):
    """
    Construct a feature table for all Wimbledon matches in test_years.
    Each row has feature differences and a binary label (1 if player A won).
    """
    rng = np.random.default_rng(seed)
    rows = []

    matches = matches.copy()
    matches['tourney_date'] = pd.to_datetime(matches['tourney_date'])
    serve_history = build_serve_history(matches)

    for year in test_years:
        wim = matches[
            (matches['tourney_name'] == 'Wimbledon') &
            (matches['year'] == year)
        ].copy()

        if len(wim) == 0:
            continue

        cutoff = wim['tourney_date'].min()

        # Train Elo on all matches strictly before this Wimbledon
        train = matches[matches['tourney_date'] < cutoff].copy()
        _, ratings = compute_elo(train)
        overall_elo = ratings['overall']
        grass_elo = ratings['grass']

        for _, match in wim.iterrows():
            w = match['winner_id']
            l = match['loser_id']

            # --- Elo difference (winner minus loser) ---
            w_blend = blend_w * grass_elo.get(w, 1500) + (1 - blend_w) * overall_elo.get(w, 1500)
            l_blend = blend_w * grass_elo.get(l, 1500) + (1 - blend_w) * overall_elo.get(l, 1500)
            elo_diff = w_blend - l_blend

            # --- Serve difference ---
            w_serve = grass_serve_rating(serve_history, w, cutoff)
            l_serve = grass_serve_rating(serve_history, l, cutoff)
            if w_serve and l_serve:
                serve_diff = w_serve['serve_points_won'] - l_serve['serve_points_won']
            else:
                serve_diff = 0.0  # neutral if either player lacks grass serve data

            # --- Recent form difference ---
            w_form = recent_grass_form(matches, w, cutoff)
            l_form = recent_grass_form(matches, l, cutoff)
            if w_form and l_form:
                form_diff = w_form['recent_grass_winrate'] - l_form['recent_grass_winrate']
            else:
                form_diff = 0.0

            # --- Randomly flip so label is balanced ---
            if rng.random() < 0.5:
                # Keep orientation: player A = winner, label = 1
                rows.append({
                    'year': year,
                    'elo_diff': elo_diff,
                    'serve_diff': serve_diff,
                    'form_diff': form_diff,
                    'label': 1
                })
            else:
                # Flip orientation: player A = loser, label = 0
                rows.append({
                    'year': year,
                    'elo_diff': -elo_diff,
                    'serve_diff': -serve_diff,
                    'form_diff': -form_diff,
                    'label': 0
                })

    return pd.DataFrame(rows)


# 2. TRAIN AND EVALUATE
# We train on earlier years and test on later years (no overlap).
# We compare three models:
#   A. Elo only (elo_diff)
#   B. Elo + serve (elo_diff, serve_diff)
#   C. Elo + serve + form (all three)
# All evaluated with log loss, Brier score, and accuracy.

def evaluate_models(features, train_years, test_years):
    """Train logistic models on train_years, evaluate on test_years."""
    train = features[features['year'].isin(train_years)]
    test = features[features['year'].isin(test_years)]

    print(f"Train: {len(train)} matches ({min(train_years)}-{max(train_years)})")
    print(f"Test:  {len(test)} matches ({min(test_years)}-{max(test_years)})")

    feature_sets = {
        'A. Elo only': ['elo_diff'],
        'B. Elo + serve': ['elo_diff', 'serve_diff'],
        'C. Elo + serve + form': ['elo_diff', 'serve_diff', 'form_diff'],
    }

    print(f"\n{'Model':<25} {'LogLoss':>9} {'Brier':>8} {'Acc':>7}")
    print("-" * 52)

    from sklearn.preprocessing import StandardScaler

    results = {}
    for name, cols in feature_sets.items():
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train[cols])
        test_scaled = scaler.transform(test[cols])

        model = LogisticRegression()
        model.fit(train_scaled, train['label'])

        prob = model.predict_proba(test_scaled)[:, 1]
        pred = (prob > 0.5).astype(int)

        ll = log_loss(test['label'], prob, labels=[0, 1])
        bs = brier_score_loss(test['label'], prob)
        acc = (pred == test['label']).mean()

        results[name] = {'log_loss': ll, 'brier': bs, 'acc': acc, 'model': model, 'cols': cols}
        print(f"{name:<25} {ll:>9.4f} {bs:>8.4f} {acc:>7.3f}")

    return results


# 3. MAIN

if __name__ == '__main__':
    DATA_DIR = 'data/raw/tennis_atp'

    # Load cached feature table if it exists, otherwise build it
    if os.path.exists('outputs/feature_table.csv'):
        print("Loading cached feature table...")
        features = pd.read_csv('outputs/feature_table.csv')
    else:
        print("Loading matches...")
        matches = load_matches(DATA_DIR, start_year=2000)
        print("Building feature table (this takes a few minutes)...")
        features = build_feature_table(matches, test_years=range(2010, 2026))
        os.makedirs('outputs', exist_ok=True)
        features.to_csv('outputs/feature_table.csv', index=False)

    print(f"\nFeature table: {len(features)} matches")

    train_years = list(range(2010, 2020))
    test_years = [2021, 2022, 2023, 2024, 2025]

    results = evaluate_models(features, train_years, test_years)