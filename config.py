import os

IMG_SHAPE = (48, 48, 3)

BATCH_SIZE = 64
EPOCHS = 100

BASE_OUTPUT = "output"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])