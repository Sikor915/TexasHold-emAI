import pandas as pd
import numpy as np
import re

def parse_poker_game(game_text, player_name):
    data = {
        "num_players": 0,
        "player_stack": None,
        "player_cards": [],
        "pre_flop": [],
        "flop": [],
        "decision_flop": [],
        "turn": None,
        "decision_turn": [],
        "river": None,
        "decision_river": [],
        "net_result": 0.0,
        "winners": []
    }

    # Extract players and stacks
    seats = re.findall(r"Seat \d+: (\w+) \(([\d\.]+)\)", game_text)
    data["num_players"] = len(seats)
    for player, stack in seats:
        if player == player_name:
            data["player_stack"] = float(stack)

    # Cards dealt to player
    data["player_cards"] = re.findall(rf"Player {player_name} received card: \[(\w\w)\]", game_text)

    # Fold tracking
    folded_stage = None
    fold_match = re.search(rf"Player {player_name} folds", game_text)
    if fold_match:
        fold_pos = fold_match.start()
        if fold_pos < game_text.find("*** FLOP ***"):
            folded_stage = "pre-flop"
        elif fold_pos < game_text.find("*** TURN ***"):
            folded_stage = "flop"
        elif fold_pos < game_text.find("*** RIVER ***"):
            folded_stage = "turn"
        else:
            folded_stage = "river"

    # Pre-flop actions
    pre_flop_section = game_text.split("*** FLOP ***")[0]
    data["pre_flop"] = re.findall(rf"Player {player_name} (\w+)(?: \([\d\.]+\))?", pre_flop_section)

    # Flop, turn, river cards
    flop_match = re.search(r"\*\*\* FLOP \*\*\*: \[(.*?)\]", game_text)
    if flop_match:
        data["flop"] = flop_match.group(1).split()

    turn_match = re.search(r"\*\*\* TURN \*\*\*: \[.*?\] \[(\w\w)\]", game_text)
    if turn_match:
        data["turn"] = turn_match.group(1)

    river_match = re.search(r"\*\*\* RIVER \*\*\*: \[.*?\] \[(\w\w)\]", game_text)
    if river_match:
        data["river"] = river_match.group(1)

    # Flop decisions
    if folded_stage is None or folded_stage != "pre-flop":
        flop_section = re.search(r"\*\*\* FLOP \*\*\*(.*?)(\*\*\* TURN \*\*\*|Game ended at:)", game_text, re.DOTALL)
        if flop_section:
            data["decision_flop"] = re.findall(rf"Player {player_name} (\w+)", flop_section.group(1))
        if folded_stage == "flop":
            data["decision_flop"].append("folded")

    # Turn decisions
    if folded_stage not in ["pre-flop", "flop"]:
        turn_section = re.search(r"\*\*\* TURN \*\*\*(.*?)(\*\*\* RIVER \*\*\*|Game ended at:)", game_text, re.DOTALL)
        if turn_section:
            data["decision_turn"] = re.findall(rf"Player {player_name} (\w+)", turn_section.group(1))
        if folded_stage == "turn":
            data["decision_turn"].append("folded")

    # River decisions
    if folded_stage not in ["pre-flop", "flop", "turn"]:
        river_section = re.search(r"\*\*\* RIVER \*\*\*(.*?)(------|Game ended at:)", game_text, re.DOTALL)
        if river_section:
            data["decision_river"] = re.findall(rf"Player {player_name} (\w+)", river_section.group(1))
        if folded_stage == "river":
            data["decision_river"].append("folded")

    # Net result
    summary_line = re.search(rf"Player {player_name} .*?Bets: ([\d\.]+).*?Collects: ([\d\.]+)", game_text)
    if summary_line:
        bet = float(summary_line.group(1))
        collected = float(summary_line.group(2))
        data["net_result"] = round(collected - bet, 2)

    # Winners
    data["winners"] = re.findall(r"\*Player (\w+)", game_text)

    return data


def parse_all_games_from_file(file_path, player_name, output_csv):
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split by each game
    games = re.findall(r"Game started at:.*?Game ended at:.*?(?=Game started at:|$)", full_text, re.DOTALL)
    print(f"Found {len(games)} games.")

    parsed_data = []
    for i, game_text in enumerate(games):
        game_data = parse_poker_game(game_text, player_name)
        parsed_data.append(game_data)

    # Prepare DataFrame for CSV
    df = pd.DataFrame(parsed_data)
    df["player_cards"] = df["player_cards"].apply(lambda x: " ".join(x))
    df["flop"] = df["flop"].apply(lambda x: " ".join(x))
    df["pre_flop"] = df["pre_flop"].apply(lambda x: ", ".join(x))
    df["decision_flop"] = df["decision_flop"].apply(lambda x: ", ".join(x))
    df["decision_turn"] = df["decision_turn"].apply(lambda x: ", ".join(x))
    df["decision_river"] = df["decision_river"].apply(lambda x: ", ".join(x))
    df["winners"] = df["winners"].apply(lambda x: ", ".join(x))

    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")

# Example usage:
player_name = "IlxxxlI"

test_file_path = "../../KaggleDataSet/test.txt"
output_csv_test = "../../KaggleDataSet/parsed_test.csv"

file_path_01 = "../../KaggleDataSet/Export Holdem Manager 2.0.txt"
output_csv_01 = "../../KaggleDataSet/parsed_poker_games_2.0.csv"

file_path_02 = "../../KaggleDataSet/Export Holdem Manager 2.1.txt"
output_csv_02 = "../../KaggleDataSet/parsed_poker_games_2.1.csv"

#parse_all_games_from_file(test_file_path, player_name, output_csv_test)
parse_all_games_from_file(file_path_01, player_name, output_csv_01)
parse_all_games_from_file(file_path_02, player_name, output_csv_02)
