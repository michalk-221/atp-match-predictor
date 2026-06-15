# ATP Match Predictor

Predikce výsledků ATP tenisových zápasů pomocí strojového učení. Model trénovaný na 25 000+ zápasech z období 2016–2026.

## Výsledky

| Model | Přesnost |
|---|---|
| Baseline (vždy výše postavený hráč) | 64.2 % |
| Model bez kurzů | 65.1 % |
| **Model s kurzy** | **68.8 %** |

## Klíčové zjištění

Sázkové kurzy jsou nejsilnějším prediktorem výsledku zápasu. Žádná další veřejně dostupná informace (povrch kurtu, head-to-head statistika, věk hráče, kolo turnaje, indoor/outdoor) nepřidala statisticky významnou hodnotu nad rámec rankingu a kurzů. Kurzy od bookmakerů tyto faktory již implicitně obsahují – jedná se o praktickou aplikaci hypotézy efektivního trhu na sportovní sázení.

## Struktura projektu

```
tennis_predictor/
├── train.py                       # trénování a uložení modelů
├── app.py                         # Streamlit frontend
├── README.md
└── .gitignore
```

Po spuštění `train.py` se vytvoří:

```
├── tennis_model_with_odds.pkl     # model s kurzy (68.8 %)
├── tennis_model_without_odds.pkl  # model bez kurzů (65.1 %)
└── player_ranks.pkl               # poslední rankingy hráčů
```

## Instalace

```bash
pip install pandas numpy scikit-learn joblib streamlit kagglehub
```

Pro stažení dat přes Kaggle API je nutný účet na [kaggle.com](https://www.kaggle.com) a nastavený API token (`~/.kaggle/kaggle.json`).

## Použití

**1. Natrénuj modely**

```bash
python train.py
```

Skript automaticky stáhne potřebné datasety z Kaggle, natrénuje oba modely a uloží je jako `.pkl` soubory.

**2. Spusť aplikaci**

```bash
streamlit run app.py
```

Otevři prohlížeč na `http://localhost:8501`, vyber dva hráče a volitelně zadej sázkové kurzy z libovolné sázkové kanceláře.

## Metodologie

**Features:**
- `Rank_diff` – rozdíl ATP rankingů obou hráčů
- `Implied_prob_1` – pravděpodobnost výhry odvozená ze sázkového kurzu (1 / kurz)

**Model:** Logistická regrese trénovaná na zápasech 2016–2026 s chronologickým rozdělením dat (80 % trénink / 20 % test). Chronologické dělení zabraňuje data leakage, který by vznikl náhodným rozdělením u časových řad.

**Head-to-head:** H2H statistiky jsou počítány výhradně z historicky předcházejících zápasů, aby nedocházelo k úniku informací z budoucnosti.

## Licence a attribution

Kód v tomto repozitáři je pod licencí MIT.

Data pocházejí z:
- [ATP Tennis 2000–2026](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) – výsledky zápasů, rankingy, kurzy
- [Huge Tennis Database](https://www.kaggle.com/datasets/guillemservera/tennis) – odvozeno z díla [Jeffa Sackmanna](https://github.com/JeffSackmann), licencováno pod [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Raw data nejsou součástí tohoto repozitáře. Stáhni je ručně z výše uvedených odkazů nebo automaticky přes `train.py`.
