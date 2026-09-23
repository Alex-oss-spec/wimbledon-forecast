"""Score the pre-tournament 2026 forecast against what actually happened.

The forecast in outputs/wimbledon_2026_forecast.csv was committed on
29 June 2026, before main draw play began. This script leaves it untouched
and compares it with the real results in results/actual_2026.csv
(every player who reached the last 16, and how far they went).

For each round it reports:
  - Brier score and log loss for the model
  - the same for a no-information baseline (every player equally likely)
  - Brier skill score (1 = perfect, 0 = no better than the baseline)
  - how many of the model's top-k picks actually got there
"""
import unicodedata
import numpy as np
import pandas as pd

ROUNDS = [("p_r16", "R16", 16), ("p_qf", "QF", 8), ("p_sf", "SF", 4),
          ("p_final", "F", 2), ("p_win", "W", 1)]
ORDER = ["R16", "QF", "SF", "F", "W"]


def key(name):
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()


def main():
    fc = pd.read_csv("outputs/wimbledon_2026_forecast.csv")
    actual = pd.read_csv("results/actual_2026.csv")
    reached = {key(r.player_name): ORDER.index(r.furthest_round) for r in actual.itertuples()}
    fc["k"] = fc["player_name"].map(key)
    missing = set(reached) - set(fc["k"])
    assert not missing, f"names not matched to the forecast: {missing}"

    rows = []
    for col, label, n in ROUNDS:
        y = fc["k"].map(lambda k: float(reached.get(k, -1) >= ORDER.index(label))).values
        p = fc[col].clip(1e-6, 1 - 1e-6).values
        base = n / len(fc)
        brier, brier0 = np.mean((p - y) ** 2), np.mean((base - y) ** 2)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        ll0 = -np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))
        top = fc.nlargest(n, col)["k"].map(lambda k: reached.get(k, -1) >= ORDER.index(label)).sum()
        rows.append({"round": label, "players": n, "brier": round(brier, 4),
                     "brier_baseline": round(brier0, 4), "brier_skill": round(1 - brier / brier0, 2),
                     "log_loss": round(ll, 3), "log_loss_baseline": round(ll0, 3),
                     "top_picks_correct": f"{top}/{n}"})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("results/scorecard_2026.csv", index=False)

    # Biggest surprises: players who went further than the model thought likely.
    print("\nLowest pre-tournament probability of reaching the round they reached:")
    for r in actual.itertuples():
        col = dict((l, c) for c, l, _ in ROUNDS)[r.furthest_round]
        p = fc.loc[fc["k"] == key(r.player_name), col].iloc[0]
        if p < 0.05:
            print(f"  {r.player_name}: {r.furthest_round} at {p:.2%}")


if __name__ == "__main__":
    main()
