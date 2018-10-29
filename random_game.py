from game.board import Board, random_move, score
from game.ominos import Transformation

b = Board()

while not b.game_over:
    player = b.next_player
    omino_idx, transformation, x, y = random_move(b, player)
    b.place(player, omino_idx, transformation, x, y)
    print(b)

print(b)
print('Game over!  Scores:')
for player in range(1, 5):
    print('{}: {}'.format(player, score(b, player)))
