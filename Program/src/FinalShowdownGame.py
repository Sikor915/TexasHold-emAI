from DQN_AI.DeepLearningNetwork import load_or_initialize_model, DQNPokerAgent
from DQN_AI.PokerGame import DQNPokerPlayer

from pypokerengine.api.game import setup_config, start_poker
from GeneticPlayerStuff.GeneticAgent import GeneticPlayer
from GeneticPlayerStuff.Helper_Functions import get_best_probability, get_best_aggresion

from LearnedBotPlayer.LearnedBot import LearnedBot
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

PLAYERS = 6
GAMES = 10
ROUNDS_PER_GAME = 5
STARTING_STACK = 4000
SMALL_BLIND = 100

if __name__ == "__main__":
    config = setup_config(max_round=ROUNDS_PER_GAME, initial_stack=STARTING_STACK, small_blind_amount=SMALL_BLIND)

    Genetic_Player = GeneticPlayer(get_best_probability(), get_best_aggresion())
    config.register_player(name="Genetic Player1", algorithm=Genetic_Player)
    config.register_player(name="Genetic Player2", algorithm=Genetic_Player)

    dqn_player_model = load_or_initialize_model("./my_dqn_model.keras", (150,), 3)
    dqn_player_agent = DQNPokerAgent(150, 3, model=dqn_player_model, epsilon=0.0)

    DQN_Player = DQNPokerPlayer(dqn_player_agent, "DQN Player", 150, False)
    config.register_player(name="DQN Player1", algorithm=DQN_Player)
    config.register_player(name="DQN Player2", algorithm=DQN_Player)

    learned_bot = LearnedBot(model_path="poker_model.pkl", epsilon=0.0)
    config.register_player(name="Learned Bot1", algorithm=learned_bot)
    config.register_player(name="Learned Bot2", algorithm=learned_bot)

    # Init dictionary
    player_names = ["Genetic Player1", "Genetic Player2",
                     "DQN Player1", "DQN Player2",
                       "Learned Bot1", "Learned Bot2"] 
    player_totals = {name: 0 for name in player_names}

    for i in range(GAMES):
        game_result = start_poker(config, verbose=0)
        print(f"Game {i + 1}/{GAMES} finished. Results: {game_result['players']}")
        
        # Update players sum
        for player in game_result["players"]:
            player_name = player["name"]
            player_stack = player["stack"]
            player_totals[player_name] += player_stack - STARTING_STACK

    # Summary
    print("\nSummary after all games:")
    for player_name, total in player_totals.items():
        print(f"{player_name}: {total} $ (around {total/GAMES:.2f} per game)")