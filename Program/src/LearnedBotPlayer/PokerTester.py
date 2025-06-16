import os
import pandas as pd
import joblib
import eval7
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Configuration
MODEL_PATH = "poker_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_FOLDER = 'PluribusDataSet'

# Code cards
rank_dict = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
suit_dict = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

def encode_card(card_str):
    if len(card_str) == 2:
        return rank_dict.get(card_str[0], 0), suit_dict.get(card_str[1], 0)
    return 0, 0

def encode_cards(cards):
    parts = cards.strip().split()
    return [encode_card(c) for c in parts] + [(0,0)] * (5 - len(parts))

# Poker calcs
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

# Get all files
def load_text_files(folder_path):
    """Ładuje wszystkie pliki .txt z folderu i zwraca połączoną zawartość"""
    all_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                all_texts.append(file.read())
    return all_texts

def parse_hand_text(text, player_name):
    try:
        # Num of players
        num_players_match = re.search(r'Table .* (\d+)-max', text)
        num_players = int(num_players_match.group(1)) if num_players_match else 6

        # Get a certain player stack
        stack_pattern = re.compile(rf'Seat \d+: {re.escape(player_name)} \(([\d\.]+) in chips\)')
        stack_match = stack_pattern.search(text)
        if not stack_match:
            return None
        player_stack = float(stack_match.group(1)) * 100

        # Player cards
        hole_cards_pattern = re.compile(rf'Dealt to {re.escape(player_name)} \[([2-9TJQKA][shdc]) ([2-9TJQKA][shdc])\]')
        hole_cards_match = hole_cards_pattern.search(text)
        if not hole_cards_match:
            return None
        player_cards = f"{hole_cards_match.group(1)} {hole_cards_match.group(2)}"

        # Flop, turn, river
        flop_match = re.search(r'\*\*\* FLOP \*\*\* \[([2-9TJQKA][shdc]) ([2-9TJQKA][shdc]) ([2-9TJQKA][shdc])\]', text)
        flop = " ".join(flop_match.groups()) if flop_match else ""

        turn_match = re.search(r'\*\*\* TURN \*\*\* \[[^\]]+\] \[([2-9TJQKA][shdc])\]', text)
        turn = turn_match.group(1) if turn_match else ""

        river_match = re.search(r'\*\*\* RIVER \*\*\* \[[^\]]+\] \[([2-9TJQKA][shdc])\]', text)
        river = river_match.group(1) if river_match else ""

        # Find decision based on river
        decision = None
        river_section_index = text.find("*** RIVER ***")
        if river_section_index != -1:
            lines_after_river = text[river_section_index:].split("\n")
            for line in lines_after_river:
                line = line.strip()
                if line.startswith(f"{player_name}:"):
                    tokens = line.split()
                    if len(tokens) > 1 and tokens[1] in ["bets", "calls", "raises", "folds", "checks"]:
                        decision = tokens[1]
                        break

        if decision is None:
            return None  # No decision

        p_rank1, p_suit1 = encode_card(player_cards.split()[0])
        p_rank2, p_suit2 = encode_card(player_cards.split()[1])

        f_ranks_suits = [encode_card(c) for c in flop.split()] if flop else [(0, 0)] * 3
        while len(f_ranks_suits) < 3:
            f_ranks_suits.append((0, 0))

        t_rank, t_suit = encode_card(turn) if turn else (0, 0)
        r_rank, r_suit = encode_card(river) if river else (0, 0)

        return {
            "num_players": num_players,
            "player_stack": player_stack,
            "player_cards": player_cards,
            "flop": flop,
            "turn": turn,
            "river": river,
            "decision_river": decision,
            "decision": decision,
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
        print(f"Error parsing hand text for player {player_name}: {e}")
        return None

def get_players_in_hand(text):
    # Search for all players from game
    pattern = re.compile(r'Seat \d+: (\w+) \([\d\.]+ in chips\)')
    return pattern.findall(text)
    

def predict_decision(cards_str, flop_str, turn_str, river_str, num_players, stack, model, label_encoder):
    """Przewiduje decyzję na podstawie aktualnego stanu gry"""
    board = " ".join(filter(None, [flop_str, turn_str, river_str]))

    hand_strength = compute_hand_strength(cards_str, board)
    equity = calc_equity_vs_random(cards_str, board)
    pot_size = stack * 0.5
    other_players_no = num_players - 1

    p_rank1, p_suit1 = encode_card(cards_str.split()[0])
    p_rank2, p_suit2 = encode_card(cards_str.split()[1])
    
    f_ranks_suits = [encode_card(c) for c in flop_str.split()] if flop_str else [(0,0)]*3
    while len(f_ranks_suits) < 3:
        f_ranks_suits.append((0,0))
        
    t_rank, t_suit = encode_card(turn_str) if turn_str else (0,0)
    r_rank, r_suit = encode_card(river_str) if river_str else (0,0)

    # Prepare input data
    input_data = [
        num_players, stack, pot_size, other_players_no,
        hand_strength, equity, equity,
        p_rank1, p_suit1, p_rank2, p_suit2,
        f_ranks_suits[0][0], f_ranks_suits[0][1],
        f_ranks_suits[1][0], f_ranks_suits[1][1],
        f_ranks_suits[2][0], f_ranks_suits[2][1],
        t_rank, t_suit, r_rank, r_suit
    ]
    
    # Prediction
    prediction = model.predict([input_data])[0]
    return label_encoder.inverse_transform([prediction])[0]

if __name__ == "__main__":
    # Read from csv files
    df1 = pd.read_csv("KaggleDataSet/parsed_poker_games_2.0.csv")
    df2 = pd.read_csv("KaggleDataSet/parsed_poker_games_2.1.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    
    df = df[df["decision_river"].notnull()]
    df["decision"] = df["decision_river"].apply(lambda x: x.split(",")[0].strip())

    # Read from txt files
    hand_texts = load_text_files(DATA_FOLDER)
    parsed_data = []
    
    hand_texts = load_text_files(DATA_FOLDER)
    parsed_data = []

    for text in hand_texts:
        hands = re.split(r'(?=PokerStars Hand #)', text)
        for hand in hands:
            if hand.strip():
                players = get_players_in_hand(hand)
                for player in players:
                    parsed_hand = parse_hand_text(hand, player)
                    if parsed_hand:
                        parsed_data.append(parsed_hand)

    df_txt = pd.DataFrame(parsed_data)


    # Prepare data for training
    label_encoder = LabelEncoder()
    all_decisions = pd.concat([df["decision"], df_txt["decision"]]).unique()
    label_encoder.fit(all_decisions)

    df["label"] = label_encoder.transform(df["decision"])
    if not df_txt.empty:
        df_txt["label"] = df_txt["decision"].apply(
            lambda d: label_encoder.transform([d])[0] if d in label_encoder.classes_ else label_encoder.transform(["check"])[0]
        )

    for df_part in [df, df_txt]:
        df_part["board"] = df_part[["flop", "turn", "river"]].fillna("").agg(" ".join, axis=1).str.strip()
        df_part["hand_strength"] = df_part.apply(
            lambda row: compute_hand_strength(row["player_cards"], row["board"]), axis=1)
        df_part["equity_vs_random"] = df_part.apply(
            lambda row: calc_equity_vs_random(row["player_cards"], row["board"]), axis=1)
        df_part["pot_size"] = df_part["player_stack"] * 0.5
        df_part["other_players_no"] = df_part["num_players"] - 1


    # Connect all data
    df_full = pd.concat([df, df_txt], ignore_index=True)

    # Choose features
    features = [
        "num_players", "player_stack", "pot_size", "other_players_no",
        "hand_strength", "equity_vs_random", "equity_vs_random",
        "p_rank1", "p_suit1", "p_rank2", "p_suit2",
        "f_rank1", "f_suit1", "f_rank2", "f_suit2", "f_rank3", "f_suit3",
        "t_rank", "t_suit", "r_rank", "r_suit"
    ]

    X_full = df_full[features]
    y_full = df_full["label"]

    df_full["pot_size"] = df_full["player_stack"] * 0.5
    df_full["other_players_no"] = df_full["num_players"] - 1


    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"Precision on testing model: {model.score(X_test, y_test):.4f}")

    # Save model and encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    # Example of prediction
    model = joblib.load("poker_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Call predict_decision with all required arguments
    decision = predict_decision(
        cards_str="As Ks",
        flop_str="Js 7s 2h",
        turn_str="6s",
        river_str="9d",
        num_players=6,
        stack=45,
        model=model,
        label_encoder=label_encoder
    )
    print("Predicted decision", decision)