import argparse

from google.cloud import firestore

from game.game import Game
from game.ominos import Transformation

parser = argparse.ArgumentParser(description='Download a game from d4ab firestore.')
parser.add_argument('game_id', help='firebase event list id, eg A1lsjyMEoF14IkuznfKo')

args = parser.parse_args()

db = firestore.Client()
doc = db.document('games', args.game_id).get()

g = Game()

events = doc.to_dict()['events']
for e in events:
    if e['type'] != 'place':
        continue

    t = Transformation(int(e['currentTransformation']['rotations']), int(e['currentTransformation']['flips']))
    g.play_move((int(e['playerIndex']), int(e['selectedOminoIdx']), t, int(e['cell']['i']), int(e['cell']['j'])))

print(g)
