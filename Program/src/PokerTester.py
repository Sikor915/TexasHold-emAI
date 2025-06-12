import pandas as pd
import joblib
import eval7
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# === ŚCIEŻKI ===
MODEL_PATH = "poker_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

# === KODOWANIE KART ===
rank_dict = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
suit_dict = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

def encode_card(card_str):
    if len(card_str) == 2:
        return rank_dict.get(card_str[0], 0), suit_dict.get(card_str[1], 0)
    return 0, 0

def encode_cards(cards):
    parts = cards.strip().split()
    return [encode_card(c) for c in parts] + [(0,0)] * (5 - len(parts))

# === OBLICZ MOC RĘKI ===
def compute_hand_strength(player_cards, board_cards):
    try:
        cards = encode_cards(player_cards)[:2] + encode_cards(board_cards)[:5]
        eval_cards = [eval7.Card("23456789TJQKA"[r] + "shdc"[s]) for r, s in cards]
        while len(eval_cards) < 7:
            eval_cards.append(eval7.Card("2c"))  # dummy
        score = eval7.evaluate(eval_cards)
        return 1 - (score / 7462)
    except:
        return 0

# === OBLICZ EQUITY ===
def calc_equity_vs_random(hole_str, board_str, iters=1000):
    try:
        deck = eval7.Deck()
        hole = [eval7.Card(c) for c in hole_str.split()]
        board = [eval7.Card(c) for c in board_str.split() if c != ""]

        for card in hole + board:
            deck.cards.remove(card)

        wins = 0
        for _ in range(iters):
            deck.shuffle()
            opp_hand = deck.peek(2)
            full_board = board + deck.draw(5 - len(board))

            our_score = eval7.evaluate(hole + full_board)
            opp_score = eval7.evaluate(opp_hand + full_board)

            if our_score > opp_score:
                wins += 1
            elif our_score == opp_score:
                wins += 0.5

        return wins / iters
    except:
        return 0

# === WCZYTYWANIE DANYCH CSV ===
df1 = pd.read_csv("KaggleDataSet/parsed_poker_games_2.0.csv")
df2 = pd.read_csv("KaggleDataSet/parsed_poker_games_2.1.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Używamy decyzji na riverze i enkodujemy etykiety
df = df[df["decision_river"].notnull()]
df["decision"] = df["decision_river"].apply(lambda x: x.split(",")[0].strip())

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["decision"])

# Kodowanie kart w df
df[['p_rank1', 'p_suit1', 'p_rank2', 'p_suit2']] = df['player_cards'].fillna("").apply(
    lambda x: pd.Series([*sum(encode_cards(x)[:2], ())])
)
df[['f_rank1', 'f_suit1', 'f_rank2', 'f_suit2', 'f_rank3', 'f_suit3']] = df['flop'].fillna("").apply(
    lambda x: pd.Series([*sum(encode_cards(x)[:3], ())])
)
df[['t_rank', 't_suit']] = df['turn'].fillna("").apply(
    lambda x: pd.Series(encode_card(x.strip()) if x.strip() else (0,0))
)
df[['r_rank', 'r_suit']] = df['river'].fillna("").apply(
    lambda x: pd.Series(encode_card(x.strip()) if x.strip() else (0,0))
)

# Obliczamy cechy
df["board"] = df[["flop", "turn", "river"]].fillna("").agg(" ".join, axis=1).str.strip()
df["hand_strength"] = df.apply(
    lambda row: compute_hand_strength(row["player_cards"], row["board"]),
    axis=1
)
df["equity_vs_random"] = df.apply(
    lambda row: calc_equity_vs_random(row["player_cards"], row["board"]),
    axis=1
)

# === PARSOWANIE TEKSTOWYCH ROZDAŃ ===
def parse_hand_text(text):
    try:
        num_players = 6
        player_stack = 10000  # możesz dostosować lub wyciągnąć z tekstu

        hole_cards_match = re.search(r'Dealt to \w+ \[([2-9TJQKA][shdc]) ([2-9TJQKA][shdc])\]', text)
        if not hole_cards_match:
            return None
        player_cards = hole_cards_match.group(1) + " " + hole_cards_match.group(2)

        flop_match = re.search(r'\*\*\* FLOP \*\*\* \[([2-9TJQKA][shdc]) ([2-9TJQKA][shdc]) ([2-9TJQKA][shdc])\]', text)
        flop = " ".join(flop_match.groups()) if flop_match else ""

        turn_match = re.search(r'\*\*\* TURN \*\*\* \[[^\]]+\] \[([2-9TJQKA][shdc])\]', text)
        turn = turn_match.group(1) if turn_match else ""

        river_match = re.search(r'\*\*\* RIVER \*\*\* \[[^\]]+\] \[([2-9TJQKA][shdc])\]', text)
        river = river_match.group(1) if river_match else ""

        # Przykładowa decyzja gracza po riverze (dopasuj swojego gracza)
        decision = None
        river_section_index = text.find("*** RIVER ***")
        if river_section_index != -1:
            lines_after_river = text[river_section_index:].split("\n")
            for line in lines_after_river:
                line = line.strip()
                if line.startswith("MrBlue:") or line.startswith("Joe:"):
                    if any(word in line for word in ["bets", "calls", "raises", "folds", "checks"]):
                        decision = line.split()[1]
                        break
        if decision is None:
            decision = "check"  # fallback

        p_rank1, p_suit1 = encode_card(player_cards.split()[0])
        p_rank2, p_suit2 = encode_card(player_cards.split()[1])

        f_ranks_suits = [(0,0)]*3
        if flop:
            f_cards = flop.split()
            f_ranks_suits = [encode_card(c) for c in f_cards]
            if len(f_ranks_suits) < 3:
                f_ranks_suits += [(0,0)]*(3-len(f_ranks_suits))

        t_rank, t_suit = encode_card(turn) if turn else (0,0)
        r_rank, r_suit = encode_card(river) if river else (0,0)

        return {
            "num_players": num_players,
            "player_stack": player_stack,
            "player_cards": player_cards,
            "flop": flop,
            "turn": turn,
            "river": river,
            "decision_river": decision,
            "p_rank1": p_rank1,
            "p_suit1": p_suit1,
            "p_rank2": p_rank2,
            "p_suit2": p_suit2,
            "f_rank1": f_ranks_suits[0][0],
            "f_suit1": f_ranks_suits[0][1],
            "f_rank2": f_ranks_suits[1][0],
            "f_suit2": f_ranks_suits[1][1],
            "f_rank3": f_ranks_suits[2][0],
            "f_suit3": f_ranks_suits[2][1],
            "t_rank": t_rank,
            "t_suit": t_suit,
            "r_rank": r_rank,
            "r_suit": r_suit,
        }
    except Exception as e:
        print(f"Error parsing hand text: {e}")
        return None

# Przykładowe rozdadanie tekstowe (wstaw swój tekst)
hand_texts = [
"""
PokerStars Hand #207205512056:  Hold'em No Limit ($0.01/$0.02 USD) - 2023/06/04 21:03:01 ET
Table 'Leif 6' 6-max Seat #4 is the button
Seat 1: MrBlue ($2.01 in chips)
Seat 2: Joe ($2.25 in chips)
Seat 3: Jack ($2.50 in chips)
Seat 4: Peter ($2.12 in chips)
Seat 5: Carol ($2.11 in chips)
Seat 6: Tom ($2.04 in chips)
Dealt to MrBlue [As Ks]
Joe: folds
Jack: calls $0.02
Peter: raises $0.06 to $0.08
Carol: folds
Tom: folds
MrBlue: calls $0.08
*** FLOP *** [Js 7s 2h]
Peter: bets $0.10
MrBlue: raises $0.25 to $0.35
Jack: folds
Peter: calls $0.25
*** TURN *** [Js 7s 2h] [6s]
Peter: checks
MrBlue: bets $0.50
Peter: folds
"""
]

parsed_data = [parse_hand_text(t) for t in hand_texts]
parsed_data = [d for d in parsed_data if d is not None]
df_txt = pd.DataFrame(parsed_data)
# Po wczytaniu CSV i parsowaniu tekstów:
print(df_txt.columns)

# Połącz decyzje z obu datasetów i fituj LabelEncoder na wszystkich
all_decisions = pd.concat([df["decision"], df_txt["decision"]]).unique()
label_encoder = LabelEncoder()
label_encoder.fit(all_decisions)

# Zamień na liczby
df["label"] = label_encoder.transform(df["decision"])
df_txt["label"] = label_encoder.transform(df_txt["decision"])

# Dodaj kolumnę "decision"
df_txt["decision"] = df_txt["decision_river"].apply(lambda x: x.split(",")[0].strip())

# Sprawdź i zakoduj etykiety tekstowe względem label_encoder
known_classes = set(label_encoder.classes_)
unknown_labels = set(df_txt["decision"]) - known_classes
if unknown_labels:
    print(f"Nowe etykiety w danych tekstowych: {unknown_labels}. Zamieniam na 'check'.")
df_txt["label"] = df_txt["decision"].apply(lambda d: label_encoder.transform([d])[0] if d in known_classes else label_encoder.transform(["check"])[0])

# Oblicz cechy
df_txt["board"] = df_txt[["flop", "turn", "river"]].fillna("").agg(" ".join, axis=1).str.strip()
df_txt["hand_strength"] = df_txt.apply(lambda r: compute_hand_strength(r["player_cards"], r["board"]), axis=1)
df_txt["equity_vs_random"] = df_txt.apply(lambda r: calc_equity_vs_random(r["player_cards"], r["board"]), axis=1)

# Cecha modelu
features = [
    "num_players", "player_stack",
    "hand_strength", "equity_vs_random",
    "p_rank1", "p_suit1", "p_rank2", "p_suit2",
    "f_rank1", "f_suit1", "f_rank2", "f_suit2", "f_rank3", "f_suit3",
    "t_rank", "t_suit", "r_rank", "r_suit"
]

# Połącz dane CSV i tekstowe i trenuj model
df_full = pd.concat([df, df_txt], ignore_index=True)
X_full = df_full[features]
y_full = df_full["label"]

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"🎯 Dokładność na testowym zbiorze: {model.score(X_test, y_test):.4f}")
print("🧠 Możliwe decyzje:", list(label_encoder.classes_))

# Zapis modelu i label_encoder
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

# === FUNKCJA PREDYKCJI ===
def predict_decision(cards_str, flop_str, turn_str, river_str, num_players, stack):
    board = " ".join(filter(None, [flop_str, turn_str, river_str]))
    hand_strength = compute_hand_strength(cards_str, board)
    equity = calc_equity_vs_random(cards_str, board)

    encoded_cards = encode_cards(cards_str)[:2]
    encoded_flop = encode_cards(flop_str)[:3]
    encoded_turn = encode_card(turn_str.strip()) if turn_str else (0,0)
    encoded_river = encode_card(river_str.strip()) if river_str else (0,0)

    encoded_input = [
        num_players, stack,
        hand_strength, equity,
        *sum(encoded_cards, ()),
        *sum(encoded_flop, ()),
        *encoded_turn,
        *encoded_river
    ]

    encoded = model.predict([encoded_input])[0]
    return label_encoder.inverse_transform([encoded])[0]

# === PRZYKŁAD ===
decision = predict_decision(
    cards_str="As Ks",
    flop_str="Js 7s 2h",
    turn_str="6s",
    river_str="9d",
    num_players=6,
    stack=45
)
print("🤖 Przewidywana decyzja:", decision)
