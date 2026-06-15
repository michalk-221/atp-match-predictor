import joblib
import numpy as np
import streamlit as st

model_with_odds    = joblib.load("tennis_model_with_odds.pkl")
model_without_odds = joblib.load("tennis_model_without_odds.pkl")
player_ranks       = joblib.load("player_ranks.pkl")

player_list = sorted(player_ranks.keys())

st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾")
st.title("🎾 ATP Match Predictor")
st.caption("Model trénovaný na 25 000+ ATP zápasech (2016–2026)")

# ── Výběr hráčů ──────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hráč 1")
    player1 = st.selectbox("Vyber hráče 1", player_list, key="p1")
    rank1 = player_ranks.get(player1)
    st.metric("Ranking", f"#{int(rank1)}" if rank1 else "Neznámý")

with col2:
    st.subheader("Hráč 2")
    player2 = st.selectbox("Vyber hráče 2", player_list, key="p2")
    rank2 = player_ranks.get(player2)
    st.metric("Ranking", f"#{int(rank2)}" if rank2 else "Neznámý")

# ── Sázkové kurzy (nepovinné) ─────────────────────────────────────────────────

st.divider()
st.subheader("Sázkové kurzy (nepovinné)")
st.caption("Zadáním kurzů zvýšíš přesnost z 65.1 % na 68.8 %")

col3, col4 = st.columns(2)
with col3:
    odd1 = st.number_input(
        f"Kurz – {player1}",
        min_value=0.0,
        value=0.0,
        step=0.05,
        key="odd1",
        help="Nech 0 pokud kurz nemáš"
    )
with col4:
    odd2 = st.number_input(
        f"Kurz – {player2}",
        min_value=0.0,
        value=0.0,
        step=0.05,
        key="odd2",
        help="Nech 0 pokud kurz nemáš"
    )

# ── Predikce ──────────────────────────────────────────────────────────────────

st.divider()

if st.button("Predikovat", use_container_width=True, type="primary"):
    if player1 == player2:
        st.error("Vyber dva různé hráče.")
    elif not rank1 or not rank2:
        st.error("Ranking jednoho z hráčů není znám.")
    else:
        rank_diff = rank1 - rank2
        use_odds  = odd1 > 1.0 and odd2 > 1.0

        if use_odds:
            prob = model_with_odds.predict_proba(
                np.array([[rank_diff, 1 / odd1]])
            )[0][1]
            st.info("Použit model s kurzy (přesnost 68.8 %)")
        else:
            prob = model_without_odds.predict_proba(
                np.array([[rank_diff]])
            )[0][1]
            st.info("Použit model bez kurzů (přesnost 65.1 %)")

        st.subheader("Výsledek")

        col5, col6 = st.columns(2)
        with col5:
            st.metric(
                label=player1,
                value=f"{prob * 100:.1f} %",
                delta="Favorit" if prob > 0.5 else None
            )
        with col6:
            st.metric(
                label=player2,
                value=f"{(1 - prob) * 100:.1f} %",
                delta="Favorit" if prob < 0.5 else None
            )

        st.progress(float(prob))

        winner     = player1 if prob > 0.5 else player2
        confidence = max(prob, 1 - prob)
        st.success(f"Předpokládaný vítěz: **{winner}** ({confidence * 100:.1f} %)")
