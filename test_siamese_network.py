import cv2 as cv
import numpy as np
import utils
import config
from tensorflow import keras

names = ['akatsuki', 'byakuya', 'carmine', 'chaos', 'eltnum', 'enkidu', 'gordeau', 'hilda', 'hyde', 'linne', 'londrekia', 'merkava', 'mika', 'nanase', 'orie', 'phonon', 'seth', 'vatista', 'wagner', 'waldstein', 'yuzuriha']
nametocode = { names[i] : i for i in range(len(names)) }

dir = 'uni_char/test/'
tmpX = []
tmpY = []
for name in names:
    for count in range(1,3):
        tmpX.append(cv.imread(dir+name+'_'+str(count)+'.png'))
        tmpY.append(nametocode[name])
tmpX = np.array(tmpX)
X_test = tmpX / 255.0
y_test = np.array(tmpY)

(pair_test, label_test) = utils.make_pairs(X_test, y_test)

model = keras.models.load_model(config.MODEL_NAME)

x_test_1 = pair_test[:, 0]
x_test_2 = pair_test[:, 1]

results = model.evaluate([x_test_1, x_test_2], label_test)
print("test loss, test acc:", results)