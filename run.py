from tqdm import tqdm
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from game.game import PLAYERS, WIDTH, HEIGHT, random_game, play_game
from player import Player, keras_evaluator

# Build the model
model = Sequential()
# Some random layers
model.add(Dense(units=64, activation='relu', input_dim=(PLAYERS * WIDTH * HEIGHT)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
# Prediction layer
model.add(Dense(units=PLAYERS, activation='softmax'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print('Generating games...')
masks = []
results = []
for _ in tqdm(range(100)):
    game = random_game()
    masks += game.masks
    results += [game.winners] * game.turns
X = np.array(masks)
Y = np.array(results)

print('Training model...')
model.fit(x=X, y=Y, epochs=50, validation_split=0.1)

# Evaluate the model
players = [Player(i, keras_evaluator(model, i)) for i in [1,2,3,4]]
game = play_game(*players)

for (s, b) in zip(game.masks, game.blobs):
    print(b)
    print('Predictions: ', model.predict(np.array([s]))[0])
    print('\n')

print(game)
print(game.winners)
random_x = np.array(game.masks)
random_y = np.array([game.winners] * game.turns)
print(model.metrics_names)
print(model.test_on_batch(random_x, random_y))
