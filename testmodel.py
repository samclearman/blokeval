from time import time

import numpy as np
import keras

from game.game import random_game


model = keras.models.load_model('models/1626639387.1357684')

print('Loaded model')

# print(model.summary())

# Evaluate the model
for _ in range(20):
  game = random_game()

  # print(game.blobs[-1])
  print('Predictions: ', model.predict(np.array([game.masks[-1]]))[0])
  print('Scores: ', game.scores)
  print('Winners: ', game.winners)
  print('\n')

  
