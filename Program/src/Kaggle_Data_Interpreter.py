import pandas as pd
import numpy as np
import re

def parse_poker_game(game_text, player_name):
    data = {
        "num_players": 0, #technically not needed
        "player_stack": None, #player's money
        "blind":0, #0 - no, 1 - small, 2 - big
        "player_cards": [], #cards in hand
        "pre_flop": [], #actions before flop
        "flop": [], #cards on flop
        "decision_flop": [], #actions on flop
        "turn": None, #card on turn
        "decision_turn": [], #actions on turn
        "river": None, #card on river
        "decision_river": [], #actions on river
        "net_result": 0.0, #net result
    }

    # Extract players and stacks
    seats = re.findall(r"Seat \d+: (\w+) \(([\d\.]+)\)", game_text)
    data["num_players"] = len(seats)
    for player, stack in seats:
        if player == player_name:
            data["player_stack"] = float(stack)

    if re.search(rf"Player {player_name} has small blind", game_text):
        data["blind"] = 1
    elif re.search(rf"Player {player_name} has big blind", game_text):
        data["blind"] = 2

    # Cards dealt to player
    data["player_cards"] = re.findall(rf"Player {player_name} received card: \[(\w{{2,3}})\]", game_text)

    # Fold tracking
    folded_stage = None
    fold_match = re.search(rf"Player {player_name} folds", game_text)
    if fold_match:
        fold_pos = fold_match.start()

        flop_pos = game_text.find("*** FLOP ***")
        turn_pos = game_text.find("*** TURN ***")
        river_pos = game_text.find("*** RIVER ***")

        if flop_pos == -1 or fold_pos < flop_pos:
            folded_stage = "pre-flop"
        elif turn_pos == -1 or fold_pos < turn_pos:
            folded_stage = "flop"
        elif river_pos == -1 or fold_pos < river_pos:
            folded_stage = "turn"
        else:
            folded_stage = "river"


    # Pre-flop actions - remove the "received" action because its useless
    if "*** FLOP ***" in game_text:
        pre_flop_section = game_text.split("*** FLOP ***")[0]
    else:
        # Fallback: cut at summary if no flop occurs
        pre_flop_section = game_text.split("------ Summary ------")[0]
    data["pre_flop"] = [
        action for action in re.findall(rf"Player {player_name} (\w+)(?: \([\d\.]+\))?", pre_flop_section)
        if action not in ["has", "posts", "received"]  # Exclude "has small/big blind" and "received a card"
    ]
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

    # Extract all game sections only if they exist
    flop_start = game_text.find("*** FLOP ***")
    turn_start = game_text.find("*** TURN ***")
    river_start = game_text.find("*** RIVER ***")
    summary_start = game_text.find("Summary")

    # Flop section
    if folded_stage is None or folded_stage != "pre-flop":
        if flop_start != -1:
            flop_end = turn_start if turn_start != -1 else summary_start
            if flop_end != -1:
                flop_text = game_text[flop_start:flop_end]
                data["decision_flop"] = re.findall(rf"Player {player_name} (\w+)", flop_text)

    # Turn section
    if folded_stage not in ["pre-flop", "flop"]:
        if turn_start != -1:
            turn_end = river_start if river_start != -1 else summary_start
            if turn_end != -1:
                turn_text = game_text[turn_start:turn_end]
                data["decision_turn"] = re.findall(rf"Player {player_name} (\w+)", turn_text)

    # River section
    if folded_stage not in ["pre-flop", "flop", "turn"]:
        if river_start != -1:
            river_end = summary_start if summary_start != -1 else len(game_text)
            river_text = game_text[river_start:river_end]
            data["decision_river"] = re.findall(rf"Player {player_name} (\w+)", river_text)


    # Net result
    summary_line = re.search(rf"Player {player_name} .*?Bets: ([\d\.]+).*?Collects: ([\d\.]+)", game_text)
    if summary_line:
        bet = float(summary_line.group(1))
        collected = float(summary_line.group(2))
        data["net_result"] = round(collected - bet, 2)

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

parse_all_games_from_file(test_file_path, player_name, output_csv_test)
parse_all_games_from_file(file_path_01, player_name, output_csv_01)
parse_all_games_from_file(file_path_02, player_name, output_csv_02)
