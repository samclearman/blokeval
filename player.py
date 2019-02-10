import numpy as np
from game.game import flat_mask
from game.board import Cell, valid_moves, place_cells

def keras_evaluator(model, player):
    cells = None
    turn = 100
    def f(board, move):
        nonlocal cells, turn
        if cells == None or len(board.ominos_remaining[player]) != turn:
            turn = len(board.ominos_remaining[player])
            cells = [Cell(*c) for c in board.cells]
        place_cells(cells, player, *move)
        predictions = model.predict(np.array([flat_mask(cells)]))
        return predictions[0][player - 1]
    return f

# Right now this is a basic player which just picks whichever move has the best evaluator score
class Player:
    def __init__(self, player_idx, evaluator):
        self._evaluator = evaluator
        self._player_idx = player_idx

    def next_move(self, game):
        b = game.board
        if b.next_player != self._player_idx:
            raise 'Not my turn'
        best_score = 0
        move = None
        for move in valid_moves(b, self._player_idx):
            score = self._evaluator(b, move)
            if score > best_score:
                move = move
                best_score = score
        return (self._player_idx, ) + move

