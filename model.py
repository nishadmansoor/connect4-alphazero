"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board():
    """Return a 6x7 integer numpy array of zeros representing an empty Connect-4 board."""
    # TODO: create a 6x7 integer array of zeros and return it
    return np.zeros((6,7), dtype=int)

# Step 2 - column_top_row
def column_top_row(board, column):
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    # TODO: scan the column from the bottom up and return the first empty row index
    for row in range(5,-1,-1):
        if board[row,column] == 0:
            return row
    return -1

# Step 3 - drop_piece
def drop_piece(board, column, player):
    # TODO: place `player` in the lowest empty row of `column` and return the new board
    row =  column_top_row(board,column)
    #print(row)

    new_board = board.copy()
    if row == -1:
        raise ValueError("column is full")
    else:
       new_board[row,column] = player

    return new_board

# Step 4 - column_full
import numpy as np

def column_full(board, column):
    """Return True if `column` has no empty rows left."""
    # TODO: check whether the column can still accept a piece
    if column_top_row(board,column) == -1:
        return True
    else:
        return False

# Step 5 - valid_moves
def valid_moves(board):
    # TODO: return a list of column indices that still have at least one empty row
    space = []

    for col in range(7):
        if board[0,col] == 0:
            space.append(col)
    return space

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board):
    # TODO: scan every row for four consecutive matching non-zero pieces horizontally
    for row in range(6):
        for col in range(4):
            if board[row,col] != 0 and board[row,col] == board[row,col + 1] == board[row,col + 2] == board[row,col + 3]:
                return board[row,col]
    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board):
    # TODO: scan every column for four consecutive matching non-zero pieces vertically
        for col in range(7):
            for row in range(3):
                if board[row,col] != 0 and board[row,col] == board[row + 1,col] == board[row + 2,col] == board[row + 3,col]:
                    return board[row,col]
        return 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board):
    # TODO: scan every down-right diagonal of the 6x7 board for four matching non-zero pieces
    for row in range(3):
        for col in range(4):
            if board[row,col] != 0 and board[row,col] == board[row + 1,col + 1] == board[row + 2,col + 2] == board[row + 3,col + 3]:
                return board[row,col]
    return 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board):
    # TODO: scan every up-right diagonal for four consecutive matching non-zero pieces
    for row in range(3,6):
        for col in range(4):
            if board[row,col] != 0 and board[row,col] == board[row - 1, col + 1] == board[row - 2, col + 2] == board[row - 3, col + 3]:
                return board[row,col]
    return 0

# Step 10 - check_winner
import numpy as np

def check_winner(board):
    """Return 1 or 2 if that player has four in a row, else 0."""
    # TODO: combine the four direction scans and return the first non-zero result
    check = [
        four_in_a_row_horizontal(board),
        four_in_a_row_vertical(board),
        four_in_a_row_diagonal_down_right(board),
        four_in_a_row_diagonal_up_right(board)
    ]

    for winner in check:
        if winner != 0:
            return winner
    return 0

# Step 11 - board_is_full
def board_is_full(board):
    # TODO: return True when no column has an empty slot left
    return len(valid_moves(board)) == 0

# Step 12 - is_terminal
def is_terminal(board):
    # TODO: return (done, winner) using check_winner and board_is_full.
    winner = int(check_winner(board))

    if winner != 0:
        return True, winner
    if board_is_full(board):
        return True, 0
    return False,0

# Step 13 - other_player
def other_player(player):
    # TODO: return the opponent's player code (1 <-> 2)
    return 3 - player

# Step 14 - step_env
def step_env(board, column, player):
    # TODO: drop piece for player, then return (new_board, done, winner, next_player).
    new_board = drop_piece(board,column,player)
    done, winner = is_terminal(new_board)
    next_player = other_player(player)

    return new_board, done, winner, next_player

# Step 15 - encode_board
import numpy as np

def encode_board(board, current_player):
    """Encode a 6x7 board as a (2, 6, 7) float32 tensor from current_player's view."""
    opp = other_player(current_player)
    
    player_plane = (board == current_player).astype(np.float32)
    opponent_plane = (board == opp).astype(np.float32)
    
    return np.stack([player_plane, opponent_plane], axis=0)

# Step 16 - board_to_torch_tensor
import torch
def board_to_torch_tensor(board, current_player):
    # TODO: encode the board and return it as a float32 torch tensor of shape (1, 2, 6, 7).
    encode = encode_board(board, current_player)
    tensor = torch.tensor(encode, dtype=torch.float32)

    return tensor.unsqueeze(0)

# Step 17 - init_conv_backbone
import torch
import torch.nn as nn
def init_conv_backbone(in_channels=2, hidden_channels=16):
    # TODO: Build a small convolutional backbone preserving the 6x7 spatial shape.
    
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels,hidden_channels,kernel_size=3,padding=1),
        nn.ReLU()
    )

# Step 18 - init_policy_head
import torch
import torch.nn as nn

def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    # TODO: build a small policy head that projects backbone features to column logits
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 2, kernel_size=1),
        nn.BatchNorm2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2 * 6 * 7, num_columns)
    )

# Step 19 - init_value_head
import torch
import torch.nn as nn

def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    # TODO: build a value head that collapses backbone features to a single bounded scalar per board.
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.BatchNorm2d(1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1 * 6 * 7, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Tanh()
    )

# Step 20 - build_policy_value_net
import torch
import torch.nn as nn

def build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7):
    """Compose backbone + policy head + value head into one nn.Module."""
    # TODO: build an nn.Module with backbone, policy_head, value_head attributes
    net = nn.Module()
    net.backbone = init_conv_backbone(in_channels=in_channels, hidden_channels=hidden_channels)
    net.policy_head = init_policy_head(hidden_channels=hidden_channels, num_columns=num_columns)
    net.value_head = init_value_head(hidden_channels=hidden_channels)
    
    def forward(x):
        feats = net.backbone(x)
        return net.policy_head(feats), net.value_head(feats)
    
    net.forward = forward
    return net

# Step 21 - policy_value_forward
import torch
import torch.nn as nn

def policy_value_forward(net, encoded_board):
    """Run encoded_board (B,2,6,7) through net and return (logits, value)."""
    # TODO: call the network on the encoded board and return its two outputs
    return net(encoded_board)

# Step 22 - action_mask
import numpy as np

def action_mask(board):
    # TODO: return a length-7 boolean mask, True where the column is legal
    mask = np.zeros(7, dtype=bool)
    valid = valid_moves(board)
    mask[valid] = True
    return mask

# Step 23 - masked_policy_logits
import torch

def masked_policy_logits(logits, mask):
    """Set logits at illegal columns to -inf.

    logits: torch.Tensor of shape (..., 7)
    mask:   bool array/tensor of shape (7,), True = legal
    returns: torch.Tensor of same shape as logits
    """
    # TODO: replace logits at illegal columns with negative infinity
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool, device=logits.device)
    else:
        mask = mask.to(device=logits.device, dtype=torch.bool)
    
    return torch.where(mask, logits, float('-inf'))

# Step 24 - masked_log_softmax
import torch

def masked_log_softmax(logits, mask):
    """Log-softmax of logits with illegal columns (mask=False) forced to -inf."""
    # TODO: mask out illegal columns, then apply log-softmax over the last dim.
    masked_logits = masked_policy_logits(logits, mask)
    return torch.log_softmax(masked_logits, dim=-1)

# Step 25 - sample_action_from_policy
import torch

def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""
    # TODO: scale logits by temperature, mask illegal columns, sample one index
    masked_logits = masked_policy_logits(logits / temperature, mask)
    distance = torch.distributions.Categorical(logits=masked_logits)
    return distance.sample().item()

# Step 26 - greedy_action_from_policy
import torch

def greedy_action_from_policy(logits, mask):
    """Return the argmax legal column index from masked policy logits."""
    # TODO: mask out illegal columns then return the argmax as a python int
    masked_logits = masked_policy_logits(logits, mask)

    return torch.argmax(masked_logits).item()

# Step 27 - make_mcts_node
def make_mcts_node(prior=0.0, parent=None):
    # TODO: build a dict with prior, visit_count, value_sum, children, and parent fields.
    return {
        'prior': prior,
        'visit_count': 0,
        'value_sum': 0.0,
        'children': {},
        'parent': parent
    }

# Step 28 - node_q_value (not yet solved)
# TODO: implement

# Step 29 - ucb_score (not yet solved)
# TODO: implement

# Step 30 - select_best_child (not yet solved)
# TODO: implement

# Step 31 - select_leaf (not yet solved)
# TODO: implement

# Step 32 - evaluate_with_network (not yet solved)
# TODO: implement

# Step 33 - expand_node (not yet solved)
# TODO: implement

# Step 34 - backup_value (not yet solved)
# TODO: implement

# Step 35 - run_one_simulation (not yet solved)
# TODO: implement

# Step 36 - run_mcts (not yet solved)
# TODO: implement

# Step 37 - visit_count_policy (not yet solved)
# TODO: implement

# Step 38 - mcts_choose_action (not yet solved)
# TODO: implement

# Step 39 - record_self_play_step (not yet solved)
# TODO: implement

# Step 40 - play_self_play_game (not yet solved)
# TODO: implement

# Step 41 - assign_value_targets (not yet solved)
# TODO: implement

# Step 42 - generate_self_play_batch (not yet solved)
# TODO: implement

# Step 43 - value_loss_mse (not yet solved)
# TODO: implement

# Step 44 - policy_loss_cross_entropy (not yet solved)
# TODO: implement

# Step 45 - l2_regularization_loss (not yet solved)
# TODO: implement

# Step 46 - combined_loss (not yet solved)
# TODO: implement

# Step 47 - encode_batch_states (not yet solved)
# TODO: implement

# Step 48 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 49 - training_step (not yet solved)
# TODO: implement

# Step 50 - training_epoch (not yet solved)
# TODO: implement

# Step 51 - self_play_iteration (not yet solved)
# TODO: implement

# Step 52 - train_loop (not yet solved)
# TODO: implement

# Step 53 - random_policy_action (not yet solved)
# TODO: implement

# Step 54 - greedy_agent_action (not yet solved)
# TODO: implement

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

