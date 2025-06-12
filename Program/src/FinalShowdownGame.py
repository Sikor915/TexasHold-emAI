import PokerGame

from DeepLearningNetwork import load_or_initialize_model, DQNPokerAgent
from PokerGame import DQNPokerPlayer
from pypokerengine.api.game import setup_config, start_poker
from GeneticPlayerStuff.Genetic_Agent import GeneticPlayer
from GeneticPlayerStuff.Helper_Functions import get_best_probability, get_best_aggresion



PLAYERS = 2
GAMES = 1000
ROUNDS_PER_GAME = 5
STARTING_STACK = 10000
SMALL_BLIND = 100



if __name__ == "__main__":
    config = setup_config(ROUNDS_PER_GAME, STARTING_STACK, SMALL_BLIND)

    Genetic_Player = GeneticPlayer(get_best_probability(), get_best_aggresion())
    config.register_player(name="Genetic Player", algorithm=Genetic_Player)
    #config.register_player(name="Genetic Player2", algorithm=Genetic_Player)

    dqn_player_model = load_or_initialize_model("./my_dqn_model.keras", (150,), 3)
    dqn_player_agent = DQNPokerAgent(150, 3, model= dqn_player_model, epsilon=0.0)

    DQN_Player = DQNPokerPlayer(dqn_player_agent, "DQN Player", 150, False)
    config.register_player(name="DQN Player", algorithm=DQN_Player)
    #config.register_player(name="DQN Player2", algorithm=DQN_Player)

    for i in range(GAMES):
        game_result = start_poker(config, verbose=0)
        print(f"Game {i + 1}/{GAMES} finished. Results: {game_result['players']}")