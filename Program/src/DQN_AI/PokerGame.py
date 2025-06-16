import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

from IPython.display import clear_output
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from DQN_AI.DeepLearningNetwork import load_or_initialize_model, DQNPokerAgent, encode_card, compute_reward

log_dir = os.path.join(os.path.dirname(__file__), "Logs")
os.makedirs(log_dir, exist_ok=True)

# Constants
NUM_SUITS = 4
NUM_RANKS = 13
MAX_COMMUNITY_CARDS = 5
LARGE_NUMBER = 10000
NUM_PLAYERS = 4
BETTING_HISTORY_SIZE = 20  # Last 20 actions

default_category = 0

def estimate_hand_strength(nb_simulation, nb_player, hole_card, community_card):
    simulation_results = []
    for i in range(nb_simulation):
        opponents_cards = []
        for j in range(nb_player-1):  # nb_opponents = nb_player - 1
            opponents_cards.append(draw_cards_from_deck(num=2))
        nb_need_community = 5 - len(community_card)
        community_card.append(draw_cards_from_deck(num=nb_need_community))
        result = observe_game_result(hole_card, community_card, opponents_cards)  # return 1 if win else 0
        simulation_results.append(result)
    average_win_rate = 1.0 * sum(simulation_results) / len(simulation_results)
    return average_win_rate

def setup_poker_game(dqn_agent1, initial_stack=1000):
    input_shape = (150,)
    action_size = 3
    state_size = input_shape[0]

    config = setup_config(max_round=100, initial_stack=initial_stack, small_blind_amount=100)

    # Register the main training player
    dqn_player1 = DQNPokerPlayer(dqn_agent1, "DQN_Player_Train", state_size, training_mode=True)
    config.register_player(name="DQN_Player_Train", algorithm=dqn_player1)

    # Register other players, potentially older versions or different strategies
    for i in range(1, NUM_PLAYERS):
        model_path = './my_dqn_model' + str(i) + '.keras'
        dqn_model = load_or_initialize_model(model_path, input_shape, action_size)
        dqn_agent = DQNPokerAgent(state_size, action_size, model=dqn_model, epsilon=1.0)
        dqn_player = DQNPokerPlayer(dqn_agent1, f"DQN_Player_Train_{i}", state_size, training_mode=True)
        config.register_player(name=f"DQN_Player_Train_{i}", algorithm=dqn_player)
        dqn_model.save(model_path)
    # # Optionally, add other types of players
    # config.register_player(name="Random_Player", algorithm=RandomPlayer())
    # config.register_player(name="SmartPlayer", algorithm=SmartPlayer())

    return config

NB_SIMULATION = 1000



class DQNPokerPlayer(BasePokerPlayer):
    def __init__(self, dqn_agent, name, state_size, training_mode=True):
        super().__init__()
        self.dqn_agent = dqn_agent
        self.name = name
        self.current_state = np.zeros(state_size)
        self.prev_state = np.zeros(state_size)
        self.hole_card = None
        self.last_action = None
        self.action_index = None
        self.bet_amount_category = None
        self.done = False
        self.last_action_details = {'action': None, 'amount': 0}  # Initialize with default values

        self.training_mode = training_mode  # New flag to indicate training mode
        self.games_since_last_learn = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.current_state is None:
            self.current_state = self._extract_state(round_state, hole_card)
            print("State size:", state.shape)  # Should output (130,)

        action, bet_amount_category = self.dqn_agent.act(self.current_state, valid_actions)
        #print(hole_card)
        #print(bet_amount_category)
        self.last_action = action
        self.bet_amount_category = bet_amount_category

        action_index = self.map_action_to_index(action, valid_actions)
        self.action_index = action_index

        action_dict = next((item for item in valid_actions if item["action"] == action), None)

        if action == 'raise' and isinstance(action_dict['amount'], dict):
            #print(action_dict['amount'])
            min_raise = action_dict['amount']['min']
            max_raise = action_dict['amount']['max']
            fraction = (self.bet_amount_category + 1) / 20
            amount = int(min_raise + fraction * (max_raise - min_raise))
        else:
            amount = action_dict['amount'] if action_dict else 0
        #         print('## amount', amount)

        self.last_action_details = {'action': action, 'amount': amount}  # Update action details
        #         print(f"declare_action - last_action_details: {self.last_action_details}")

        return action, amount

    def receive_game_start_message(self, game_info):
        for seat in game_info['seats']:
            if seat['name'] == self.name:
                self.uuid = seat['uuid']
                break

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card  # Update hole_card at the start of each round

    def receive_game_update_message(self, action, round_state):
        self.prev_state = self.current_state  # Store the current state as the previous state

        if self.current_state is None:
            print('None detected in receive_game_update_message', self.current_state)

        if self.hole_card:  # Ensure hole_card has been set
            self.current_state = self._extract_state(round_state, self.hole_card)

        self.pot_size_before = round_state['pot']['main']['amount']
        self.stack_size_before = self._get_stack_size(self.uuid, round_state)

    def _get_stack_size(self, player_uuid, round_state):
        # Iterate through the list of players (seats) in the round_state
        for player_info in round_state['seats']:
            # Check if the current player's UUID matches the specified player_uuid
            if player_info['uuid'] == player_uuid:
                # Return the stack size of the matching player
                return player_info['stack']
        # Return None or an appropriate default value if the player is not found
        return None

    def receive_round_result_message(self, winners, hand_info, round_state):
        is_winner = any(winner['uuid'] == self.uuid for winner in winners)

        # Use the stored action details directly, ensuring it's not None
        action_details = self.last_action_details if self.last_action_details is not None else {'action': 'default',
                                                                                                'amount': 0}

        # Adjust to handle cards that might already be strings
        hole_cards = [card if isinstance(card, str) else card.to_str() for card in self.hole_card]
        community_cards = [card if isinstance(card, str) else card.to_str() for card in round_state['community_card']]

        pot_size_before = self.pot_size_before
        pot_size_after = round_state['pot']['main']['amount']
        stack_size_before = self.stack_size_before
        stack_size_after = self._get_stack_size(self.uuid, round_state)

        self.done = True

        if self.bet_amount_category is None:
            self.bet_amount_category = 0  # Assuming '0' as default category

        if self.training_mode:

            # Compute the reward with additional parameters
            reward = compute_reward(
                round_state,
                self.last_action,
                action_details,
                is_winner,
                pot_size_before,
                pot_size_after,
                stack_size_before,
                stack_size_after,
                hole_cards,
                community_cards,
                # Number of players
            )

            self.dqn_agent.replay_buffer.add(self.prev_state, self.action_index, self.bet_amount_category, reward,
                                             self.current_state, self.done)
            self.games_since_last_learn += 1

            if self.games_since_last_learn >= 10:
                self.dqn_agent.learn()
                self.games_since_last_learn = 0

        # Reset for the next round
        self.done = False
        self.prev_state = None
        self.action_index = None
        self.bet_amount_category = None
        self.last_action_details = {'action': None, 'amount': 0}

    def _extract_state(self, round_state, hole_card):
        MAX_COMMUNITY_CARDS = 5
        MAX_PLAYERS = 10  # Define the maximum number of players you expect in any game
        LARGE_NUMBER = 10000  # Normalization factor for large values like stacks and pots

        # Encoding hole and community cards
        hole_cards_encoded = [encode_card(card) for card in hole_card]
        community_cards_encoded = [encode_card(card) for card in round_state['community_card']]

        # Determine the vector size for a single card
        N = len(
            encode_card(hole_card[0])) if hole_card else 26  # Example fallback size if encode_card size is not standard

        # Flatten encoded cards into single vectors
        hole_cards_vector = np.concatenate(hole_cards_encoded) if hole_cards_encoded else np.zeros(N * 2)
        community_cards_vector = np.concatenate(community_cards_encoded) if community_cards_encoded else np.zeros(
            N * len(community_cards_encoded))

        # Define a zero vector for padding community cards
        zero_vector = np.zeros(N)
        padded_community_cards = np.concatenate(
            [community_cards_vector] + [zero_vector for _ in range(MAX_COMMUNITY_CARDS - len(community_cards_encoded))])

        # Additional game features
        pot_size = np.array([round_state['pot']['main']['amount'] / LARGE_NUMBER])  # Normalize pot size

        # Encoding player stack sizes and positions
        seats = round_state['seats']
        stack_sizes = [seat['stack'] / LARGE_NUMBER for seat in seats if 'stack' in seat]
        positions = [1 if i == round_state.get('dealer_btn') else 0 for i in range(len(seats))]

        # Optional: Number of active players (can be useful in strategy decision making)
        active_players = [1 if 'folded' not in seat['state'] else 0 for seat in seats]

        # Padding for player-related features if there are fewer players than MAX_PLAYERS
        stack_sizes += [0] * (MAX_PLAYERS - len(stack_sizes))
        positions += [0] * (MAX_PLAYERS - len(positions))
        active_players += [0] * (MAX_PLAYERS - len(active_players))

        # Convert lists to numpy arrays
        stack_sizes = np.array(stack_sizes)
        positions = np.array(positions)
        active_players = np.array(active_players)

        # Concatenate all parts to form the full state vector
        state = np.concatenate(
            [hole_cards_vector, padded_community_cards, pot_size, stack_sizes, positions, active_players])

        return state

    def map_action_to_index(self, action, valid_actions):
        # Convert the action to an index. This implementation depends on how your actions are structured.
        action_dict = {act['action']: idx for idx, act in enumerate(valid_actions)}
        return action_dict.get(action, -1)  # Return -1 or another value for invalid actions

    def receive_street_start_message(self, street, round_state):
        pass

if __name__ == "__main__":

    # Initialize dictionaries to store metrics for each player
    total_gains = {}
    cumulative_rewards = {}
    win_rates = {}

    # Define model path
    model_path = './my_dqn_model.keras'

    # Define the input shape and action size for your model
    input_shape = (150,)
    action_size = 3
    state_size = input_shape[0]

    # Initialize or load the main model
    dqn_model = load_or_initialize_model(model_path, input_shape, action_size)
    dqn_agent = DQNPokerAgent(state_size, action_size, model=dqn_model,epsilon = 1)
    # Save the current model for the other players
    dqn_model.save(model_path)

    open(os.path.join(os.path.dirname(__file__), "Logs\\modelDataLogs.txt"), "a").write("----------------" + model_path + datetime.datetime.now().strftime(" %a %d-%b-%Y %H:%M") + "----------------\n")

    for episode in range(10000):
        initial_stack = 100000  # every game will have different initial stacks for now small blind remains 10
        # Setup and start the poker game with the current model playing against its previous version
        config = setup_poker_game(dqn_agent, initial_stack=initial_stack)
        game_result = start_poker(config, verbose=0)

        # Update metrics based on game results...
        for player in game_result['players']:
            name = player['name']
            stack_change = player['stack'] - initial_stack

            # Initialize player metrics if new
            if name not in total_gains:
                total_gains[name] = [0]
                cumulative_rewards[name] = []
                win_rates[name] = []

            # Update player metrics
            total_gains[name].append(total_gains[name][-1] + stack_change)
            cumulative_rewards[name].append(total_gains[name][-1])
            win_rate = sum(r > 0 for r in total_gains[name]) / len(total_gains[name])
            win_rates[name].append(win_rate)

            if episode % 100 == 0 and episode > 0:
                open(os.path.join(os.path.dirname(__file__), "Logs/modelDataLogs.txt"), "a").write(f"Epsilon after episode {episode}: {dqn_agent.epsilon}\n")
        """if episode % 1000 == 0 and episode > 0:
            clear_output(wait=True)
            print(f"Epsilon after episode {episode}: {dqn_agent.epsilon}")
    
            # Save the current model
            dqn_model.save(model_path)
            # Clear the current figure to ensure old plots are not shown
            plt.clf()
    
            # Plot metrics
            plt.figure(figsize=(12, 6))
    
            # Plot cumulative rewards
            plt.subplot(1, 2, 1)
            for name, rewards in cumulative_rewards.items():
                plt.plot(rewards, label=f'{name} Total Gains')
            plt.xlabel('Episode')
            plt.ylabel('Total Gains')
            plt.title('Total Gains per Episode')
            plt.legend()
    
            # Plot win rates
            plt.subplot(1, 2, 2)
            for name, rates in win_rates.items():
                plt.plot(rates, label=f'{name} Win Rate')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.title('Win Rate per Episode')
            plt.legend()
    
            plt.tight_layout()
            plt.show()"""

    open(os.path.join(os.path.dirname(__file__), "Logs/modelDataLogs.txt"), "a").write("\n----------------\n")

    clear_output(wait=True)
    print(f"Epsilon after episode {episode}: {dqn_agent.epsilon}")

    # Save the current model
    dqn_model.save(model_path)
    # Clear the current figure to ensure old plots are not shown
    plt.clf()

    # Plot metrics
    plt.figure(figsize=(12, 6))

    # Plot cumulative rewards
    plt.subplot(1, 2, 1)
    for name, rewards in cumulative_rewards.items():
        plt.plot(rewards, label=f'{name} Total Gains')
    plt.xlabel('Episode')
    plt.ylabel('Total Gains')
    plt.title('Total Gains per Episode')
    plt.legend()

    # Plot win rates
    plt.subplot(1, 2, 2)
    for name, rates in win_rates.items():
        plt.plot(rates, label=f'{name} Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Win Rate per Episode')
    plt.legend()

    plt.tight_layout()
    plt.show()



