import numpy
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

"""def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).reshape(x.shape[0], -1)
            y = y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            model.train()

    return num_correct / num_samples

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # Fully connected layer 1
        self.fc2 = nn.Linear(50, num_classes)  # Fully connected layer 2
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer
        return x

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(input_size=784, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(3):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device).reshape(data.shape[0], -1)  # Flatten images

        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_acc = check_accuracy(train_loader, model)
test_acc = check_accuracy(test_loader, model)
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")"""

pokerTrainDF = pd.read_csv("../../KaggleDataSet/parsed_poker_games_2.1.csv")
#print(pokerTrainDF.head())
print(pokerTrainDF.columns)

def encode_cards(cards_str):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['h', 'd', 's', 'c']
    card2id = {f"{r}{s}": i for i, (r, s) in enumerate([(r, s) for r in ranks for s in suits])}

    if not isinstance(cards_str, str):
        return []

    cards = cards_str.strip().split()
    return [card2id[c] for c in cards if c in card2id]

def encode_actions(actions_str):
    action_vocab = ["folds", "calls", "raises", "checks", "allin" ,"bets", "mucks", "received"]
    action2id = {a: i for i, a in enumerate(action_vocab)}

    if not isinstance(actions_str, str):
        return []

    actions = [a.strip() for a in actions_str.split(",")]
    return [action2id[a] for a in actions if a in action2id]

def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        items = [item[key] for item in batch]

        if isinstance(items[0], torch.Tensor) and items[0].dim() == 1:
            # Pad sequences
            collated[key] = pad_sequence(items, batch_first=True, padding_value=0)
        else:
            # Stack fixed-size tensors (e.g., scalar values)
            collated[key] = torch.stack(items)

    return collated

class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        blind = row["blind"]
        player_cards = row["player_cards"]
        flop = row["flop"]
        turn = row["turn"]
        river = row["river"]
        pre_flop = row["pre_flop"]
        flop_dec = row["decision_flop"]
        turn_dec = row["decision_turn"]
        river_dec = row["decision_river"]

        # Encode fields
        encoded_cards = encode_cards(player_cards)
        encoded_flop = encode_cards(flop)
        encoded_turn = encode_cards(turn)
        encoded_river = encode_cards(river)
        encoded_pre_flop = encode_actions(pre_flop)
        encoded_flop_dec = encode_actions(flop_dec)
        encoded_turn_dec = encode_actions(turn_dec)
        encoded_river_dec = encode_actions(river_dec)

        # Example output
        sample = {
            "num_players": torch.tensor(int(row["num_players"]), dtype=torch.int64),
            "player_stack": torch.tensor(float(row["player_stack"]), dtype=torch.float32),
            "blind": torch.tensor(int(blind), dtype=torch.int64),
            "cards": torch.tensor(encoded_cards, dtype=torch.long),
            "pre_flop": torch.tensor(encoded_pre_flop, dtype=torch.long),
            "flop": torch.tensor(encoded_flop, dtype=torch.long),
            "decision_flop": torch.tensor(encoded_flop_dec, dtype=torch.long),
            "turn": torch.tensor(encoded_turn, dtype=torch.long),
            "decision_turn": torch.tensor(encoded_turn_dec, dtype=torch.long),
            "river": torch.tensor(encoded_river, dtype=torch.long),
            "decision_river": torch.tensor(encoded_river_dec, dtype=torch.long),
            "net_result": torch.tensor(float(row["net_result"]), dtype=torch.float32)
        }
        return sample
"""
class PokerAI(nn.modules):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 64),  # Input layer
            nn.ReLU(),
            nn.Linear(64, 32),  # Hidden layer
            nn.ReLU(),
            nn.Linear(32, 2)    # Output layer (2 classes: win or lose)
        )
    def forward(self, x):
        return self.model(x)"""

pokerTrainDataSet = DataSet(pokerTrainDF)
pokerTrainDataLoader = DataLoader(pokerTrainDataSet, batch_size=2, shuffle=False, collate_fn=collate_fn)

for batch_idx, row in enumerate(pokerTrainDataLoader):
    print(f"Batch {batch_idx}:")
    print(row)
    break  # Just to see the first batch
