from time import time

from tqdm import tqdm
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Input, Reshape, Conv2D, Activation, Flatten, Dense
from keras.callbacks import TensorBoard

# import keras.backend as K
# from tensorflow.python import debug as tf_debug

from game.game import PLAYERS, WIDTH, HEIGHT, random_game, play_game
from player import Player, keras_evaluator

# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

# Resnet ish
pipe = []
first = Input(shape=(WIDTH * HEIGHT * PLAYERS, ))
shaped = Reshape((WIDTH, HEIGHT, PLAYERS))(first)
# 20x20x4
c = Conv2D(64, (7, 7), padding='same')(shaped)
pipe += [first, shaped, c]
# 20x20x64
for _ in range(8):
    c1 = Conv2D(64, (3, 3), padding='same')(pipe[-1])
    a1 = Activation('relu')(c1)
    c2 = Conv2D(64, (3, 3), padding='same')(a1)
    res = layers.add([pipe[-1], c2])
    a2 = Activation('relu')(res)
    pipe += [c1, a1, c2, res, a2]
shrink = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(pipe[-1])
# 10x10x128
act = Activation('relu')(shrink)
c = Conv2D(128, (3, 3), padding='same')(act)
# Is this really how the skip connections are supposed to work here?  The paper isn't clear
skip = Conv2D(128, (1, 1), strides=(2, 2), padding='same')(pipe[-1])
res = layers.add([skip, c])
pipe += [shrink, act, c, skip, res]
for _ in range(8):
    c1 = Conv2D(128, (3, 3), padding='same')(pipe[-1])
    a1 = Activation('relu')(c1)
    c2 = Conv2D(128, (3, 3), padding='same')(a1)
    res = layers.add([pipe[-1], c2])
    a2 = Activation('relu')(res)
    pipe += [c1, a1, c2, res, a2]
shrink = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(pipe[-1])
# 5x5x256
act = Activation('relu')(shrink)
c = Conv2D(256, (3, 3), padding='same')(act)
# Is this really how the skip connections are supposed to work here?  The paper isn't clear
skip = Conv2D(256, (1, 1), strides=(2, 2), padding='same')(pipe[-1])
res = layers.add([skip, c])
pipe += [shrink, act, c, skip, res]
for _ in range(8):
    c1 = Conv2D(256, (3, 3), padding='same')(pipe[-1])
    a1 = Activation('relu')(c1)
    c2 = Conv2D(256, (3, 3), padding='same')(a1)
    res = layers.add([pipe[-1], c2])
    a2 = Activation('relu')(res)
    pipe += [c1, a1, c2, res, a2]
f = Flatten()(pipe[-1])
d = Dense(4)(f)
act = Activation('softmax')(d)
pipe += [f, d, act]

model = Model(inputs=first, outputs = act)
print(model.summary())
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model

# Genrate training data
print('Generating games...')
masks = []
results = []
for _ in tqdm(range(1000)):
    game = random_game()
    masks += game.masks
    results += [game.winners] * game.turns
X = np.array(masks)
Y = np.array(results)
print(Y.shape)

# Balance classes
#

print('Training model...')
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
model.fit(x=X, y=Y, epochs=3, validation_split=0.1, callbacks=[tensorboard])

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

model.save('trained_model')
