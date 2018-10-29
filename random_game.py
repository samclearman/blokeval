from game.board import Board, random_move
from game.ominos import Transformation

b = Board()

while not b.game_over:
    player = b.next_player
    omino_idx, transformation, x, y = random_move(b, player)
    b.place(player, omino_idx, transformation, x, y)
    print(b)

print(b)
