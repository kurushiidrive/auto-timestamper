import os

IMG_SHAPE = (48, 48, 3)

BATCH_SIZE = 64
EPOCHS = 100
MARGIN = 1

BASE_OUTPUT = "output"

MODEL_NAME = 'siamese_model'
PLOT_NAME = 'plot.png'

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, MODEL_NAME])
MODEL_WEIGHTS_PATH = os.path.sep.join([BASE_OUTPUT, MODEL_NAME + '_weights'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, PLOT_NAME])