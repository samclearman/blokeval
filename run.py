import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from game.game import PLAYERS, WIDTH, HEIGHT, random_game

# Build the model
model = Sequential()
# Input layer
model.add(Dense(units=64, activation='relu', input_dim=(PLAYERS * WIDTH * HEIGHT)))
# Prediction layer
model.add(Dense(units=PLAYERS, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Train the model
game = random_game()
random_x = np.array(game.masks)
random_y = np.array([game.winners] * game.turns)
model.train_on_batch(random_x, random_y)

# Evaluate the model
game = random_game()

for (s, b) in zip(game.masks, game.blobs):
    print(b)
    print('Predictions: ', model.predict(np.array([s]))[0])
    print('\n')

print(game)
random_x = np.array(game.masks)
random_y = np.array([game.winners] * game.turns)
print(model.metrics_names)
print(model.test_on_batch(random_x, random_y))
