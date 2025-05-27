import os
import numpy as np
import tensorflow as tf
import random

from collections import deque
from tensorflow import keras
from tensorflow.keras import models, layers, Input
from tensorflow.keras.utils import to_categorical

TF_ENABLE_ONEDNN_OPTS=0

# Constants
NUM_SUITS = 4
NUM_RANKS = 13
MAX_COMMUNITY_CARDS = 5
LARGE_NUMBER = 10000
NUM_PLAYERS = 4  # Adjust based on your game setup
BETTING_HISTORY_SIZE = 20  # Last 10 actions, adjust as needed

default_category = 0

def encode_card(card):
    """Encode a card as a one-hot vector for its rank and suit."""
    rank, suit = card[1], card[0]
    rank_index = '23456789TJQKA'.index(rank)
    suit_index = 'SHDC'.index(suit)
    rank_one_hot = np.eye(NUM_RANKS)[rank_index]
    suit_one_hot = np.eye(NUM_SUITS)[suit_index]
    return np.concatenate([rank_one_hot, suit_one_hot])

def initialize_network(input_shape, action_size):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Shared layers
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dense(64, activation='relu')(x)

    # Action output
    action_output = layers.Dense(action_size, activation='linear', name='action_output')(x)

    # Bet amount output
    bet_amount_output = layers.Dense(5, activation='softmax', name='bet_amount_output')(x)

    # Build the model
    model = models.Model(inputs=input_layer, outputs=[action_output, bet_amount_output])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss={'action_output': 'mse', 'bet_amount_output': 'categorical_crossentropy'},
                  metrics={'action_output': 'accuracy', 'bet_amount_output': 'accuracy'})

    return model

def load_or_initialize_model(model_path, input_shape, action_size):
    print(model_path)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = keras.models.load_model(model_path)
    else:
        print("Initializing new model")
        model = initialize_network(input_shape, action_size)
    return model

def choose_action(state, model, valid_actions, epsilon):
    # Initialize default values
    best_action = None
    bet_amount_category = 0  # Default to the first category

    if not valid_actions:
        raise ValueError("No valid actions provided.")

    if np.random.rand() <= epsilon:
        # Exploration: Randomly choose a valid action
        best_action = np.random.choice([action['action'] for action in valid_actions])
        # Randomly choose a bet amount category (assuming a fixed number of categories)
        bet_amount_category = np.random.randint(0, 5)  # Adjust '5' to your number of bet amount categories
    else:
        # Exploitation: Predict Q-values for actions and bet amounts
        predictions = model.predict(state.reshape(1, -1))
        q_values = predictions[0][0]  # Assuming the first output corresponds to actions
        bet_amounts = predictions[1][0]  # Assuming the second output corresponds to bet amounts

        # Map valid actions to their indices
        action_indices = {action['action']: idx for idx, action in enumerate(valid_actions)}

        # Ensure there is a best action by selecting the first valid action as default
        if valid_actions:
            best_action = valid_actions[0]['action']

        # Choose the best action based on Q-values, considering only valid actions
        valid_q_values = [q_values[action_indices[action['action']]] for action in valid_actions]
        best_action = valid_actions[np.argmax(valid_q_values)]['action']

        # Choose the bet amount category with the highest probability
        bet_amount_category = np.argmax(bet_amounts)

    if best_action is None:
        raise ValueError("Unable to determine a best action. Check the logic and inputs.")

    return best_action, bet_amount_category


def compute_reward(round_state, action, action_details, is_winner, pot_size_before, pot_size_after, stack_size_before,
                   stack_size_after, hole_cards, community_cards):
    """
    Compute a simplified reward considering the action taken and game outcome without estimating hand strength.

    Args:
        round_state (dict): The state of the current round.
        action (str): The action taken by the agent.
        action_details (dict): Additional details about the action, such as the bet amount.
        is_winner (bool): Indicates whether the player won in this round.
        pot_size_before (int): The size of the pot before the action was taken.
        pot_size_after (int): The size of the pot after the action was taken.
        stack_size_before (int): The player's stack size before the action.
        stack_size_after (int): The player's stack size after the action.
        hole_cards (list): List of player's hole cards as string.
        community_cards (list): List of community cards on the table as string.

    Returns:
        reward (float): The computed reward.
    """
    reward = 0

    # Adjust rewards for winning or losing the round
    if is_winner:
        reward += 10  # Reward for winning
    else:
        reward -= 10  # Penalty for losing

    # Consider the action taken
    if action == 'fold':
        reward -= 5  # Discourage folding by default
    elif action == 'raise':
        bet_amount = action_details.get('amount', 0)
        pot_ratio = bet_amount / max(1, pot_size_before)  # Avoid division by zero
        reward += 5 * pot_ratio  # Reward for aggressive play when raising

    # Adjust reward based on changes in stack size, to encourage preservation of chips
    stack_change = stack_size_after - stack_size_before
    reward += stack_change / 100.0  # Small reward/penalty based on stack change

    return reward


def replay(replay_buffer, model, target_model, batch_size, gamma):
    """Sample a batch of experiences and use them to update the model with dual outputs (action and bet_amount).

    Args:
        replay_buffer (ReplayBuffer): The replay buffer to sample experiences from.
        model (tf.keras.Model): The current model.
        target_model (tf.keras.Model): The target model.
        batch_size (int): The number of experiences to sample from the buffer.
        gamma (float): The discount factor for future rewards.
    """
    # Sample a batch of experiences from the replay buffer
    states, actions, bet_amount_categories, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert lists to numpy arrays for batch processing
    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Predict the Q-values for the next states using the target model
    next_q_values_action, next_q_values_bet_amount = target_model.predict(next_states)

    # Compute the Q-value targets for actions
    targets_action = rewards + gamma * np.max(next_q_values_action, axis=1) * (1 - dones)

    # Predict the Q-values for the current states using the model
    q_values_action, q_values_bet_amount = model.predict(states)

    # Update the Q-values for the actions taken with the computed targets
    for i, action in enumerate(actions):
        q_values_action[i, action] = targets_action[i]

    # Replace None with default_category
    bet_amount_categories = np.array([cat if cat is not None else default_category for cat in bet_amount_categories],
                                     dtype='int64')

    # Convert bet_amount_categories to one-hot encoding for training the bet amount prediction part of the model
    bet_amount_categories_one_hot = to_categorical(bet_amount_categories, num_classes=q_values_bet_amount.shape[1])

    # Train the model
    model.fit(states, [q_values_action, bet_amount_categories_one_hot], epochs=1, verbose=0)


def update_target_network(main_q_network, target_q_network):
    """Update the target Q-network's weights with the main Q-network's weights.

    Args:
        main_q_network (tf.keras.Model): The main Q-network being trained.
        target_q_network (tf.keras.Model): The target Q-network used for stability.
    """
    # Get the weights from the main Q-network
    main_q_weights = main_q_network.get_weights()

    # Set the weights in the target Q-network
    target_q_network.set_weights(main_q_weights)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, bet_amount_category, reward, next_state, done):
        """Add a new experience to the buffer."""
        # Include bet_amount_category in the stored experience
        self.buffer.append((state, action, bet_amount_category, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        # Unpack bet_amount_category along with other experience components
        states, actions, bet_amount_categories, rewards, next_states, dones = zip(*batch)
        return states, actions, bet_amount_categories, rewards, next_states, dones


class DQNPokerAgent:
    def __init__(self, state_size, action_size, model=None, replay_buffer_size=50000, batch_size=32, gamma=0.95,
                 epsilon=1, epsilon_min=0.01, epsilon_decay=0.9995):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = model

        # If a model is provided, use it; otherwise, initialize a new model
        self.q_network = model if model else initialize_network((state_size,), action_size)

        # Initialize the target Q-network
        self.target_q_network = initialize_network((state_size,), action_size)
        self.update_target_network()

    def update_target_network(self):
        """Updates the target Q-network's weights."""
        update_target_network(self.q_network, self.target_q_network)

    def act(self, state, valid_actions):
        """Choose an action and a bet amount category based on the current state."""
        action, bet_amount_category = choose_action(np.reshape(state, [1, self.state_size]), self.q_network,
                                                    valid_actions, self.epsilon)
        return action, bet_amount_category

    def learn(self):
        """Trains the model using a batch of experiences from the replay buffer."""
        # Sample a batch of experiences from the replay buffer
        states, actions, bet_amount_categories, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Proceed with the rest of the training process by updating the Q-network and target Q-network
        replay(self.replay_buffer, self.q_network, self.target_q_network, self.batch_size, self.gamma)

        # Epsilon decay
        #print(self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

