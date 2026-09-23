# Wimbledon 2026 Men's Singles Forecasting Model

A pre-tournament probabilistic forecasting model for Wimbledon 2026, built using surface-aware Elo ratings and Monte Carlo tournament simulation. Generated and committed before main draw play began on 29 June 2026.

## What this does

The model predicts match outcomes and tournament advancement probabilities for all 128 players in the Wimbledon 2026 men's singles draw. Rather than just picking a winner, it produces calibrated probabilities for each player reaching the R16, QF, SF, Final, and winning the title.

## Pre-tournament forecast (generated 29 June 2026)

| Player | R16 | QF | SF | Final | Win |
|--------|-----|----|----|-------|-----|
| Jannik Sinner | 89.4% | 83.9% | 74.6% | 51.0% | 43.8% |
| Novak Djokovic | 85.2% | 77.2% | 69.7% | 37.0% | 30.1% |
| Alexander Zverev | 65.7% | 48.8% | 32.1% | 21.8% | 6.4% |
| Taylor Fritz | 35.1% | 24.1% | 14.2% | 9.0% | 2.2% |
| Ben Shelton | 55.9% | 32.9% | 20.6% | 10.6% | 2.2% |
| Alex de Minaur | 47.0% | 33.2% | 19.4% | 9.7% | 2.1% |
| Jack Draper | 27.4% | 17.7% | 9.6% | 5.7% | 1.3% |
| Tommy Paul | 48.1% | 29.4% | 7.0% | 2.3% | 1.2% |
| Arthur Fils | 27.6% | 16.9% | 10.6% | 5.4% | 1.2% |
| Daniil Medvedev | 39.1% | 24.0% | 5.6% | 2.0% | 1.0% |

Full 128-player forecast: `outputs/wimbledon_2026_forecast.csv`

## How it did (scored after the final, 12 July 2026)

The forecast above was committed before a ball was hit and has not been changed. Jannik Sinner, the model's favourite at 43.8%, won the title, beating Alexander Zverev in the final. Three of the model's top four picks to reach the semi-finals got there (Sinner, Djokovic, Zverev).

| Round | Model's top picks that got there | Brier score (model) | Brier score (no-information baseline) | Skill vs baseline |
|---|---|---|---|---|
| Last 16 | 9 of 16 | 0.082 | 0.109 | 25% |
| Quarter-finals | 5 of 8 | 0.038 | 0.059 | 36% |
| Semi-finals | 3 of 4 | 0.014 | 0.030 | 53% |
| Final | 1 of 2 | 0.008 | 0.015 | 47% |
| Winner | 1 of 1 | 0.003 | 0.008 | 58% |

The baseline gives every player an equal chance, so it is a floor rather than a serious competitor. Lower Brier scores are better.

**What it got wrong**

- **Arthur Fery**, a British wildcard, reached the semi-finals. The model gave him a 0.16% chance. A player with very few tour-level matches starts near the default rating, so the model had almost nothing to go on.
- **Ben Shelton**, the 4 seed, lost in the first round to qualifier Otto Virtanen. The model had Shelton at 86% to win that match.
- **Jack Draper** withdrew after the draw and was replaced by a lucky loser, so his slot in the forecast refers to a player who never played.
- Jan-Lennard Struff (4.0% to reach the quarter-finals), Shintaro Mochizuki and Roman Safiullin (both qualifiers) went further than the model expected.

Scoring code: `src/score_2026.py`. Results used: `results/actual_2026.csv`. Output: `results/scorecard_2026.csv`.

## Methodology

**Data:** 79,049 ATP tour-level matches (2000-2026) from the TennisMyLife database (CC BY-NC-SA 4.0)

**Elo engine:** Two ratings per player: overall Elo, updated on every match, and grass-specific Elo updated only on grass matches. FiveThirtyEight dynamic K-factor: K = 250 / (n + 5)^0.4

**Blend weight:** Overall and grass Elo blended as R = w·R_grass + (1-w)·R_overall. Optimal w=0.40 selected by grid search on training years 2010-2019, then evaluated on held-out years 2021-2025.

**Monte Carlo simulation:** 50,000 simulations of the 128-player bracket, respecting draw structure and path dependence.

## Key findings

- Surface-aware blended Elo achieves 72.4% match prediction accuracy across 15 Wimbledon editions (1,895 matches), versus 50% baseline
- Optimal blend weight is w=0.40 (60% overall, 40% grass Elo). Surface information helps, but overall Elo dominates because there are few grass matches per player
- Adding serve statistics and recent grass form produces no improvement over Elo alone. Surface-specific Elo already captures these signals
- Tournament-level Monte Carlo probabilities are well-calibrated: mean predicted and actual round-reach rates match to within 0.1 percentage points at every round from QF to winner across 15 editions

## Project structure

```
src/elo.py                     overall and grass Elo ratings from match history
src/evaluate.py                backtest on past Wimbledons and blend weight search
src/features.py, src/model.py  serve and form features (tested, no improvement over Elo)
src/simulate.py                50,000 Monte Carlo runs of the 2026 draw
src/tournament_calibration.py  checks round-reach probabilities against past editions
src/score_2026.py              scores the forecast against the real 2026 results
outputs/                       forecast, backtest and calibration results and charts
results/                       2026 results and scorecard
```

## How to run

```bash
pip install uv
uv sync
python src/elo.py           # Build Elo ratings
python src/evaluate.py      # Run backtest and blend weight search
python src/simulate.py      # Generate 2026 forecast
python src/tournament_calibration.py  # Tournament calibration
python src/score_2026.py    # Score the forecast against the 2026 results
```

## Data attribution

Match data: TennisMyLife ATP Database (https://stats.tennismylife.org), CC BY-NC-SA 4.0
