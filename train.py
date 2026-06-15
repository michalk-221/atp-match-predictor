import os
import numpy as np
import pandas as pd
import joblib
import kagglehub
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ── 1. DATA ──────────────────────────────────────────────────────────────────

print("Stahuji datasety...")
path = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
df = pd.read_csv(path + "/atp_tennis.csv")

for col in ["Odd_1", "Odd_2", "Pts_1", "Pts_2"]:
    df[col] = df[col].replace(-1, np.nan)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Načteno {len(df):,} zápasů.")


# ── 2. CÍLOVÁ PROMĚNNÁ ───────────────────────────────────────────────────────

df["Player_1_wins"] = (df["Winner"] == df["Player_1"]).astype(int)


# ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────

df["Rank_diff"] = df["Rank_1"] - df["Rank_2"]

# Head-to-head winrate (pouze z minulých zápasů)
print("Počítám H2H statistiky...")
h2h = {}
h2h_winrates = []

for _, row in df.iterrows():
    p1, p2, winner = row["Player_1"], row["Player_2"], row["Winner"]
    pair = tuple(sorted([p1, p2]))

    if pair not in h2h:
        winrate = 0.5
    else:
        total = sum(h2h[pair].values())
        winrate = h2h[pair].get(p1, 0) / total if total > 0 else 0.5

    h2h_winrates.append(winrate)

    if pair not in h2h:
        h2h[pair] = {p1: 0, p2: 0}
    if winner not in h2h[pair]:
        h2h[pair][winner] = 0
    h2h[pair][winner] += 1

df["H2H_winrate"] = h2h_winrates


# ── 4. FILTR: MODERNÍ ÉRA (2016–) ────────────────────────────────────────────

df_recent = df[df["Date"] >= "2016-01-01"].copy().reset_index(drop=True)
print(f"Zápasů od roku 2016: {len(df_recent):,}")


# ── 5. TRÉNOVÁNÍ ─────────────────────────────────────────────────────────────

def chronological_split(X, y, ratio=0.8):
    idx = int(len(X) * ratio)
    return X.iloc[:idx], X.iloc[idx:], y.iloc[:idx], y.iloc[idx:]


# Model S kurzy
df_odds = df_recent[df_recent["Odd_1"].notna() & (df_recent["Odd_1"] > 1)].copy()
df_odds["Implied_prob_1"] = 1 / df_odds["Odd_1"]

X_with = df_odds[["Rank_diff", "Implied_prob_1"]].dropna()
y_with = df_odds["Player_1_wins"][X_with.index]
X_tr, X_te, y_tr, y_te = chronological_split(X_with, y_with)

model_with_odds = LogisticRegression()
model_with_odds.fit(X_tr, y_tr)
acc_with = accuracy_score(y_te, model_with_odds.predict(X_te))

# Model BEZ kurzů
X_without = df_recent[["Rank_diff"]].dropna()
y_without = df_recent["Player_1_wins"][X_without.index]
X_tr2, X_te2, y_tr2, y_te2 = chronological_split(X_without, y_without)

model_without_odds = LogisticRegression()
model_without_odds.fit(X_tr2, y_tr2)
acc_without = accuracy_score(y_te2, model_without_odds.predict(X_te2))

baseline = (X_te["Rank_diff"] < 0).astype(int)
acc_baseline = accuracy_score(y_te, baseline)

print(f"\nVýsledky:")
print(f"  Baseline:         {acc_baseline:.1%}")
print(f"  Model bez kurzů:  {acc_without:.1%}")
print(f"  Model s kurzy:    {acc_with:.1%}")


# ── 6. ULOŽENÍ ───────────────────────────────────────────────────────────────

joblib.dump(model_with_odds,    "tennis_model_with_odds.pkl")
joblib.dump(model_without_odds, "tennis_model_without_odds.pkl")

player_ranks = df_recent.groupby("Player_1")["Rank_1"].last().to_dict()
joblib.dump(player_ranks, "player_ranks.pkl")

print(f"\nModely uloženy. Hráčů v databázi: {len(player_ranks):,}")
