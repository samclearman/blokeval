import json

from .board import Board, random_move, score
from .ominos import Transformation


def cell_values(b):
    return [c.val for c in b.cells]


def random_game():
    g = Game()
    while not g.game_over():
        b = g.board
        player = b.next_player
        omino_idx, transformation, x, y = random_move(b, player)
        g.play_move((player, omino_idx, transformation, x, y))
    return g


class Game:
    def __init__(self, moves = []):
        self._snapshots = []
        self._scores = {1: 0, 2: 0, 3: 0, 4: 0}
        self._b = Board()
        self._moves = []

        for m in moves:
            self.play_move(move)

    def __str__(self):
        status = (
            str(len(self._moves)) +
            ('<{}> '.format(self._b.next_player) if  not self.game_over() else '<☠> ') +
            ':'.join([str(self._scores[p]) for p in self._scores])
        )
        return status + '\n' + str(self._b)

    def json(self):
        return json.dumps({
            'scores': self._scores,
            'snapshots': self._snapshots
        })

    def play_move(self, move):
        self._b.place(*move)
        self._moves.append(move)
        self._snapshots.append(cell_values(self._b))
        for p in self._scores:
            self._scores[p] = score(self._b, p)

    @property
    def board(self):
        # Todo: return a copy of b
        return self._b

    def game_over(self):
        return self._b.game_over

