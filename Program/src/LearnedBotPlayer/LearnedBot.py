import numpy as np
import joblib
import eval7
from pypokerengine.players import BasePokerPlayer
from GeneticPlayerStuff.Helper_Functions import get_best_aggresion

class LearnedBot(BasePokerPlayer):
    def __init__(self, model_path="poker_model.pkl", encoder_path="label_encoder.pkl", epsilon=0.1):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None

    def declare_action(self, valid_actions, hole_card, round_state):
        try:
            # Przygotowanie danych
            community_cards = round_state['community_card']
            player_cards = [self._convert_card(c) for c in hole_card]
            player_cards_str = " ".join(player_cards)
            
            flop = " ".join([self._convert_card(c) for c in community_cards[:3]]) if len(community_cards) >=3 else ""
            turn = self._convert_card(community_cards[3]) if len(community_cards) >=4 else ""
            river = self._convert_card(community_cards[4]) if len(community_cards) >=5 else ""
            
            stack = round_state['seats'][round_state['next_player']]['stack']
            num_players = len([p for p in round_state['seats'] if p['state'] == 'participating'])
            
            # Predykcja
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.label_encoder.classes_)
            else:
                prediction = self.model.predict([self._extract_features(hole_card, round_state)])[0]
                action = self.label_encoder.inverse_transform([prediction])[0]

            # Obsługa akcji
            if action == "fold" and any(a['action'] == 'fold' for a in valid_actions):
                return "fold", 0
            elif action == "check" and any(a['action'] == 'check' for a in valid_actions):
                return "check", 0
            elif action == "call" and any(a['action'] == 'call' for a in valid_actions):
                return "call", [a['amount'] for a in valid_actions if a['action'] == 'call'][0]
            elif action == "raise" and any(a['action'] == 'raise' for a in valid_actions):
                raise_info = [a for a in valid_actions if a['action'] == 'raise'][0]['amount']
                amount = min(max(raise_info['min'], get_best_aggresion()), raise_info['max'])
                return "raise", amount
            else:
                return valid_actions[1]['action'], valid_actions[1]['amount']
                
        except Exception as e:
            print(f"Error in declare_action: {e}")
            return valid_actions[1]['action'], valid_actions[1]['amount']

    def _convert_card(self, card):
        """Konwertuje format kart z np. 'As' na 'As'"""
        return card[1].upper() + card[0].lower()

    def _extract_features(self, hole_card, round_state):
        """Ekstrakcja cech w formacie zgodnym z modelem"""
        community_cards = round_state['community_card']
        player_cards = [self._convert_card(c) for c in hole_card]
        board_cards = [self._convert_card(c) for c in community_cards]
        
        hand_strength = self._compute_hand_strength(player_cards, board_cards)
        equity = self._calc_equity(player_cards, board_cards)
        spr = self._calculate_spr(round_state)
        
        # Kodowanie kart
        p_rank1, p_suit1 = self._encode_card(player_cards[0]) if len(player_cards) > 0 else (0, 0)
        p_rank2, p_suit2 = self._encode_card(player_cards[1]) if len(player_cards) > 1 else (0, 0)
        
        f_ranks_suits = [self._encode_card(c) for c in board_cards[:3]]
        while len(f_ranks_suits) < 3:
            f_ranks_suits.append((0, 0))
            
        t_rank, t_suit = self._encode_card(board_cards[3]) if len(board_cards) >=4 else (0, 0)
        r_rank, r_suit = self._encode_card(board_cards[4]) if len(board_cards) >=5 else (0, 0)
        
        return [
            len(round_state['seats']),
            round_state['seats'][round_state['next_player']]['stack'],
            round_state['pot']['main']['amount'],
            len([p for p in round_state['seats'] if p['state'] == 'participating']) - 1,
            hand_strength,
            equity,
            spr,
            p_rank1, p_suit1, p_rank2, p_suit2,
            f_ranks_suits[0][0], f_ranks_suits[0][1],
            f_ranks_suits[1][0], f_ranks_suits[1][1],
            f_ranks_suits[2][0], f_ranks_suits[2][1],
            t_rank, t_suit, r_rank, r_suit
        ]

    def _encode_card(self, card_str):
        """Koduje kartę na wartości liczbowe"""
        rank_dict = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, 
                    '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
        suit_dict = {'s':0, 'h':1, 'd':2, 'c':3}
        if len(card_str) == 2:
            return rank_dict.get(card_str[0], 0), suit_dict.get(card_str[1], 0)
        return (0, 0)

    def _calculate_spr(self, round_state):
        """Oblicza stosunek stacku do puli (Stack-to-Pot Ratio)"""
        player_idx = round_state['next_player']
        player_stack = round_state['seats'][player_idx]['stack']
        pot_size = round_state['pot']['main']['amount']
        return player_stack / pot_size if pot_size > 0 else float('inf')

    def _compute_hand_strength(self, player_cards, board_cards):
        """Oblicza siłę ręki (0-1)"""
        try:
            evaluator = eval7.Evaluator()
            hole = [eval7.Card(c) for c in player_cards]
            board = [eval7.Card(c) for c in board_cards]
            while len(board) < 5:
                board.append(eval7.Card("2c"))
            score = evaluator.evaluate(hole, board)
            return 1 - (score / 7462)
        except:
            return 0

    def _calc_equity(self, player_cards, board_cards, iters=100):
        """Oblicza equity (0-1)"""
        try:
            evaluator = eval7.Evaluator()
            hole = [eval7.Card(c) for c in player_cards]
            board = [eval7.Card(c) for c in board_cards]
            
            deck = eval7.Deck()
            for card in hole + board:
                deck.cards.remove(card)
                
            wins = 0
            for _ in range(iters):
                deck.shuffle()
                opp_hole = deck.peek(2)
                full_board = board + deck.draw(5 - len(board))
                
                our_score = evaluator.evaluate(hole + full_board)
                opp_score = evaluator.evaluate(opp_hole + full_board)
                
                if our_score > opp_score:
                    wins += 1
                elif our_score == opp_score:
                    wins += 0.5
                    
            return wins / iters
        except:
            return 0

    # Pozostałe wymagane metody
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass