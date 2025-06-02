import deuces as d
import numpy as np
import GeneticPlayerStuff.Helper_Functions as hf
import sys

from pypokerengine.players import BasePokerPlayer
from deuces import card

suits = ['s', 'h', 'd', 'c']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

def initPlayer():
    init_def_prob = hf.get_def_probability()
    agg = np.random.uniform(0.1, 2.0)
    return GeneticPlayer(init_def_prob, agg)

class GeneticPlayer(BasePokerPlayer):
    def __init__(self, def_prob, agg=1):
        """

            Default prob is a matrix of probabilities for each action based on hand strength.

            RR <= 0.7 - Really bad hand
            0.7 < RR <= 0.9 - Bad hand
            0.9 < RR <= 1.1 - Average hand
            1.1 < RR <= 1.3 - Good hand
            1.3 < RR - Really good hand

            [fold, call, raise]

            [0.7, 0.2, 0.0],
            [0.6, 0.3, 0.1],
            [0.1, 0.8, 0.1],
            [0.0, 0.5, 0.5],
            [0.0, 0.2, 0.8]
        """
        self.aggresion = agg
        self.actions_prob = def_prob
        #self.round = 0


    def mutate(self, mutation_rate=0.1):
        self.aggresion = self.aggresion * (1 + np.random.uniform(-mutation_rate, mutation_rate))
        self.actions_prob = hf.normalize(self.actions_prob * (1 + np.random.uniform(-mutation_rate, mutation_rate, size=(5, 3))))

    def win_prob(self, your_hand, river, no_other_players, sim=10000):

        if len(river) == 0:
            if (your_hand[0][1] == your_hand[1][1]) or (your_hand[0][0] == your_hand[1][0]):
                return 0.6
            else:
                return 0.4
        else:
            all_cards = [b+a for a in suits for b in ranks]
            for card in your_hand:
                all_cards.remove(card)
            for card in river:
                all_cards.remove(card)

            win_count = 0
            num_cards = 5 - len(river) + 2 * no_other_players
            evaluator = d.Evaluator()

            for i in range(sim):
                cards_generated = np.random.choice(all_cards, num_cards, replace=False)
                j = 0
                player_hand = [d.Card.new(card) for card in your_hand]
                board_cards = [d.Card.new(card) for card in river]

                while len(board_cards) < 5:
                    board_cards.append(d.Card.new(cards_generated[j]))
                    j += 1
                hand_strength = evaluator.evaluate(player_hand, board_cards)
                best_strength = sys.maxsize
                no_best_hands = 1

                for k in range(no_other_players):
                    opp_hand = [d.Card.new(cards_generated[j]), d.Card.new(cards_generated[j+1])]
                    j += 2
                    opp_strength = evaluator.evaluate(opp_hand, board_cards)
                    if opp_strength < best_strength:
                        best_strength = opp_strength
                        no_best_hands = 1
                    elif opp_strength == best_strength:
                        no_best_hands += 1
                if best_strength > hand_strength:
                    win_count += 1
                elif best_strength == hand_strength:
                    win_count += 1 / no_best_hands

            return win_count / sim

    def declare_action(self, valid_actions, hole_card, round_state):

        player_hand = [card[1] + card[0].lower() for card in hole_card]
        river_cards = [card[1] + card[0].lower() for card in round_state['community_card']]
        player_no = round_state['next_player']

        players = len(round_state['seats'])
        other_players_no = 0
        for player in round_state['seats']:
            if player['state'] == "participating":
                other_players_no += 1

        pot = round_state['pot']['main']['amount']

        min_bet = valid_actions[1]['amount']
        stack = round_state['seats'][player_no]
        min_raise = valid_actions[2]['amount']['min']
        max_raise = valid_actions[2]['amount']['max']

        if len(river_cards) == 0:
            return ("call", min_bet)

        if min_bet == 0: # This is true when the player is first
            win_prob = self.win_prob(player_hand, river_cards, other_players_no - 1)
            rr = (win_prob * (other_players_no + 1))
            prob = list(self.actions_prob[np.argmin(abs(np.array([0.6, 0.8, 1.0, 1.2, 1.4]) - rr))])
            prob[1] = prob[0] + prob[1]
            prob[0] = 0
            #self.round += 1
        else:
            win_prob = self.win_prob(player_hand, river_cards, other_players_no)
            pot_size_odds = min_bet/(pot + min_bet)

            rr = win_prob / pot_size_odds
            prob = list(self.actions_prob[np.argmin(abs(np.array([0.6, 0.8, 1.0, 1.2, 1.4]) - rr))])

        action = np.random.choice(['fold', 'call', 'raise'], p=prob)

        if action == "call":
            return (action, min_bet)
        elif action == "raise":
            chips = pot / 3 * self.aggresion

            if chips < min_raise:
                chips = min_raise
            elif chips > max_raise:
                chips = max_raise
            else:
                chips = int(chips)

            if len(valid_actions) == 3 and (min_raise != -1 or max_raise != -1):
                return (action, chips)
            else:
                return ("call", min_bet)
        else:
            return (action, 0)

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass